from collections import defaultdict

import torch
from pytorch_lightning import LightningModule
from torchmetrics import SacreBLEUScore, TranslationEditRate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ModelInputs = dict[str, torch.Tensor]


class NMTModelConfig:
    """
    General class for model and dataset configuration.
    """

    def __init__(
        self,
        name: str,
        src: str = "en",
        tgt: str = "es",
        lang2id: dict[str, str] = {"en": "en", "es": "es"},
    ):
        # Model name
        self.name = name

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, src_lang=lang2id[src], tgt_lang=lang2id[tgt]
        )

        # Language pair
        self.src = src
        self.tgt = tgt

    def collate_fn(self, batch: list[dict]):
        data = defaultdict(list)
        for elem in batch:
            for k, v in elem.items():
                data[k].append(v)
        # Tokenize and pad
        data = self.tokenizer(
            data[self.src],
            text_target=data[self.tgt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return data


class MarianMTConfig(NMTModelConfig):
    """
    Model configuration for MarianMT models.
    """

    def __init__(self, src: str = "en", tgt: str = "es"):
        super().__init__(name=f"Helsinki-NLP/opus-mt-{src}-{tgt}", src=src, tgt=tgt)


class NLLB200Config(NMTModelConfig):
    """
    Model configuration for NLLB-200.
    """

    def __init__(self, src: str = "en", tgt: str = "es"):
        super().__init__(
            name="facebook/nllb-200-distilled-600M",
            src=src,
            tgt=tgt,
            lang2id={"en": "eng_Latn", "es": "spa_Latn"},
        )


class NMTModel(LightningModule):
    """
    PyTorch Lightning module for Huggingface NMT models.
    """

    def __init__(
        self,
        config: NMTModelConfig,
        lr: float = 2e-5,
        weight_decay: float = 1e-2,
    ):
        # Load Huggingface model
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.name)

        # Training params
        self.lr = lr
        self.weight_decay = weight_decay

        # Tokenizer and BOS token for generate if needed
        self.tokenizer = config.tokenizer
        self.extra_kwargs = {}
        if hasattr(self.tokenizer, "lang_code_to_id"):
            self.extra_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[
                self.tokenizer.tgt_lang
            ]

        # SacreBLEU metric
        self.bleu = SacreBLEUScore()
        # Translation Edit Rate metric
        self.ter = TranslationEditRate(normalize=True, lowercase=True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: ModelInputs, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: ModelInputs, batch_idx: int):
        _ = self._shared_step(batch, "val")

    def test_step(self, batch: ModelInputs, batch_idx: int):
        _ = self._shared_step(batch, "test")

    def _shared_step(self, batch: ModelInputs, split: str):
        outputs = self(**batch)
        if split != "test":
            self.log(
                f"{split}/loss",
                outputs.loss,
                prog_bar=True,
                batch_size=len(batch["input_ids"]),
            )
        # Obtain predictions and targets
        preds, target = self.decode_tokens(outputs, batch)
        # Update BLEU
        self.bleu(preds, target)
        self.log(
            f"{split}/bleu",
            self.bleu,
            prog_bar=True,
            batch_size=len(batch["input_ids"]),
            on_step=split == "train",
            on_epoch=split != "train",
        )
        # Update TER
        self.ter(preds, target)
        self.log(
            f"{split}/ter",
            self.ter,
            prog_bar=True,
            batch_size=len(batch["input_ids"]),
            on_step=split == "train",
            on_epoch=split != "train",
        )

        return outputs.loss

    def generate(self, inputs: str | list[str] | ModelInputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, list):
            inputs = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs = inputs | self.extra_kwargs
        tokens = self.model.generate(**model_inputs)

        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def predict_step(self, batch: str | list[str] | ModelInputs, batch_idx: int):
        return batch_idx, self.generate(batch)

    def decode_tokens(self, outputs, batch: ModelInputs):
        # Predictions
        preds = torch.argmax(outputs.logits, dim=-1)
        preds = preds.cpu()
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Target sentences
        target = batch["labels"].cpu()
        target = torch.where(target != -100, target, self.tokenizer.pad_token_id)
        target = self.tokenizer.batch_decode(target, skip_special_tokens=True)
        target = [[t] for t in target]

        return preds, target

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
