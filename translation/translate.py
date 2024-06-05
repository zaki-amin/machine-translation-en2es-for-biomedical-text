import os
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer

from translation.hpo.data import HPOCorpus, collate_fn

ROOT_DIR = Path(__file__).parent


def load_model(model_checkpoint):
    if model_checkpoint != "za17/helsinki-biomedical-finetuned":
        model_checkpoint = "Helsinki-NLP/opus-mt-en-es"
    # Send to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarianMTModel.from_pretrained(model_checkpoint)
    model.to(device)
    model.eval()
    torch.set_float32_matmul_precision("high")
    tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
    return device, model, tokenizer


def translate_text(
        inputs: str | list[str],
        model_checkpoint: str = None
):
    """
    Translates a string or list of strings from English to Spanish.
    :param inputs: A string or a list of strings to be translated.
    :param model_checkpoint: Path to the model checkpoint.
    """
    device, model, tokenizer = load_model(model_checkpoint)

    # Ensure inputs is a list
    if isinstance(inputs, str):
        inputs = [inputs]

    english_inputs_tensor = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        results = model.generate(english_inputs_tensor["input_ids"], max_length=512, num_beams=4)

    results = tokenizer.batch_decode(results, skip_special_tokens=True)
    return results


def translate_hpo(
        hpo_id: str,
        model_checkpoint: str,
        batch_size: int = 32,
        only_labels: bool = True
):
    """
    Translates an HPO term and all its descendants by ID and saves the
    translations as an XLSX file.
    :param hpo_id: HPO ID in the form HP:XXXXXXX.
    :param out_dir: Output directory where the result will be saved. Defaults to cwd
    :param model_checkpoint: Path to the model checkpoint.
    :param batch_size: Batch size to input into the model to speed up the inference. Defaults to 32.
    :param only_labels: If True, only translates the labels and not definitions, synonyms etc. of the HPO terms. Defaults to True.
    """
    device, model, tokenizer = load_model(model_checkpoint)

    # HPO dataset
    dataset = HPOCorpus(hpo_id, just_labels=only_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    with torch.no_grad():
        for idxs, inputs in tqdm(data_loader, desc="Translating HPO"):
            if isinstance(inputs, str):
                inputs = [inputs]

            english_inputs_tensor = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
            results = model.generate(english_inputs_tensor["input_ids"], max_length=512, num_beams=4)
            results = tokenizer.batch_decode(results, skip_special_tokens=True)
            dataset.set_trans(idxs, results)

    for i in range(len(dataset.terms)):
        if "synonym" in dataset.terms[i, "header"]:
            dataset.trans[i, "kind"] = "synonym"
        else:
            dataset.trans[i, "kind"] = dataset.terms[i, "header"]

    # Save the pairs as an Excel
    out_dir = "results/"
    dataset.save_pairs(out_dir, hpo_id + ".xlsx")

    return dataset
