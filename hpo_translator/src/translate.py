import os
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from hpo_translator.src.hpo import collate_fn, HPOCorpus
from hpo_translator.src.models import NMTModel, NMTModelConfig, MarianMTConfig

ROOT_DIR = Path(__file__).parent


def translate(
        inputs: str | list[str],
        model_checkpoint: str = None,
        config: NMTModelConfig = MarianMTConfig(),
):
    """
    Translates a string or list of strings from English to Spanish.

    Args:
        inputs: str | list[str]
            Sentences to be translated.
        model_checkpoint: str (None)
            Path to the trained model. If none is given, downloads pre-trained model
            from Hugging Face.
        config: NMTModelConfig
            Model configuration. Defaults to MarianMT.
    """
    # Model config and initialization
    torch.set_float32_matmul_precision("high")
    if model_checkpoint is None:
        model = NMTModel(config)
    else:
        model = NMTModel.load_from_checkpoint(model_checkpoint, config=config)
    model.eval()

    # Send to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with torch.no_grad():
        results = model.generate(inputs)

    return results


def translate_hpo(
        hpo_id: str,
        out_dir: str = None,
        model_checkpoint: str = Path(ROOT_DIR, "marianmt_clinical.ckpt").as_posix(),
        config: NMTModelConfig = MarianMTConfig(),
        batch_size: int = 32,
        only_labels: bool = False
):
    """
    Translates an HPO term and all its descendants by ID and saves the
    translations as an XLSX file.

    Args:
        hpo_id: str
            HPO ID in the form HP:XXXXXXX.
        out_dir: str
            Output directory where the result will be saved. Defaults to current
            working directory.
        model_checkpoint: str (None)
            Path to the trained model. If none is given, defaults to Clinical
            EN-ES MarianMT.
        config: NMTModelConfig
            Model configuration. Defaults to MarianMT.
        batch_size: int
            Batch size to input into the model to speed up the inference. Defaults to 32.
        only_labels: bool
            If True, only translates the labels and not definitions, synonyms etc. of the HPO terms. Defaults to False.
    """

    # Model configuration
    torch.set_float32_matmul_precision("high")
    model = NMTModel.load_from_checkpoint(model_checkpoint, config=config)
    model.eval()

    # Send to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # HPO dataset
    dataset = HPOCorpus(hpo_id, just_labels=only_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    with torch.no_grad():
        for idxs, inputs in tqdm(data_loader, desc="Translating HPO"):
            results = model.generate(inputs)
            dataset.set_trans(idxs, results)

    # Save the pairs as an Excel
    out_dir = os.getcwd() if out_dir is None else out_dir
    dataset.save_pairs(out_dir)

    return dataset
