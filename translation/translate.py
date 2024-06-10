from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from translation.hpo.data import HPOCorpus, collate_fn

ROOT_DIR = Path(__file__).parent


def load_model(model_checkpoint):
    """Model should have a MarianMT architecture"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarianMTModel.from_pretrained(model_checkpoint)
    model.to(device)
    model.eval()
    torch.set_float32_matmul_precision("high")
    tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
    return device, model, tokenizer


def translate_text(inputs: list[str], model_checkpoint: str):
    """
    Translates a list of strings from English to Spanish in batches.
    :param inputs: A list of strings to be translated.
    :param model_checkpoint: Path to the model checkpoint.
    """
    device, model, tokenizer = load_model(model_checkpoint)

    results = []
    for english in tqdm(inputs):
        input_ids = tokenizer.encode(english, return_tensors="pt").to(device)
        translated_tokens = model.generate(input_ids, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        results.append(translated_text)

    return results


def translate_hpo(hpo_id: str, model_checkpoint: str, batch_size: int = 32):
    """
    Translates an HPO term and all its descendants by ID and saves the translations as an XLSX file in results/
    directory.
    :param hpo_id: HPO ID in the form HP:XXXXXXX.
    :param model_checkpoint: Checkpoint name.
    :param batch_size: Batch size for model to speed up inference.

    """
    device, model, tokenizer = load_model(model_checkpoint)

    # HPO dataset
    dataset = HPOCorpus(hpo_id, just_labels=True)
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
