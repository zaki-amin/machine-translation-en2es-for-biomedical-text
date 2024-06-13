from abc import abstractmethod

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer, AutoTokenizer, \
    AutoModelForSeq2SeqLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


class TranslationModel:
    def __init__(self, checkpoint_name: str):
        self.checkpoint_name = checkpoint_name

    @abstractmethod
    def translate(self, source: str) -> str:
        """Translates a source text with the model
        :param source: the text to translate
        :return: str - the translation"""
        pass


class HelsinkiNLP(TranslationModel):
    def __init__(self):
        super().__init__("Helsinki-NLP/opus-mt-en-es")
        self.model = MarianMTModel.from_pretrained(self.checkpoint_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.checkpoint_name)

    def translate(self, source: str) -> str:
        input_ids = self.tokenizer.encode(source, return_tensors="pt")
        translated_tokens = self.model.generate(input_ids, num_beams=4, early_stopping=True)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def __str__(self):
        return "helsinki-nlp"


class FineTuned(TranslationModel):
    def __init__(self):
        super().__init__("za17/helsinki-biomedical-finetuned")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_name)

    def translate(self, source: str) -> str:
        input_ids = self.tokenizer.encode(source, return_tensors="pt")
        translated_tokens = self.model.generate(input_ids, num_beams=4, early_stopping=True)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def __str__(self):
        return "finetuned-all"


class Madlad(TranslationModel):
    def __init__(self):
        super().__init__('jbochi/madlad400-3b-mt')
        self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint_name).to(device)

    def translate(self, source: str) -> str:
        input_ids = self.tokenizer(f"<2es> {source}", max_length=1024, truncation=True,
                                   return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=1024, num_beams=4, early_stopping=True)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def __str__(self):
        return "madlad"


class NLLB(TranslationModel):
    def __init__(self, checkpoint_name: str):
        super().__init__(checkpoint_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_name)

    def translate(self, source: str) -> str:
        inputs = self.tokenizer(source, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id["spa_Latn"], max_length=1000
        )
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text


class NLLB600M(NLLB):
    def __init__(self):
        super().__init__("facebook/nllb-200-distilled-600M")

    def __str__(self):
        return "nllb-600M"


class NLLB3B(NLLB):
    def __init__(self):
        super().__init__("facebook/nllb-200-3.3B")

    def __str__(self):
        return "nllb-3B"
