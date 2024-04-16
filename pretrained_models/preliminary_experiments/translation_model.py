from abc import abstractmethod

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, MarianTokenizer, \
    MarianMTModel

using_gpu = False
device = "gpu" if using_gpu else "cpu"


class TranslationModel:
    def __init__(self, checkpoint_name: str):
        self.checkpoint_name = checkpoint_name

    @abstractmethod
    def translate(self, source: str) -> str:
        """Translates a source text with the model
        :param source: the text to translate
        :return: str - the translation"""
        pass

    def __str__(self):
        return self.checkpoint_name


class NLLBModel(TranslationModel):
    def __init__(self):
        super().__init__("facebook/nllb-200-distilled-600M")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_name)

    def translate(self, source: str) -> str:
        inputs = self.tokenizer(source, return_tensors="pt")
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id["spa_Latn"], max_length=30
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


class T5Model(TranslationModel):
    def __init__(self):
        super().__init__("google-t5/t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint_name)

    def translate(self, source: str) -> str:
        input_text = "translate English to Spanish: " + source
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512)
        outputs = self.model.generate(input_ids=input_ids, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class HelsinkiNLPModel(TranslationModel):
    def __init__(self):
        super().__init__("Helsinki-NLP/opus-mt-en-es")
        self.model = MarianMTModel.from_pretrained(self.checkpoint_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.checkpoint_name)

    def translate(self, source: str) -> str:
        input_ids = self.tokenizer.encode(source, return_tensors="pt")
        translated_tokens = self.model.generate(input_ids, num_beams=4, early_stopping=True)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text


def main():
    sentence = "The patient is suffering from cystic fibrosis."

    helsinki_nlp_model = HelsinkiNLPModel()
    print(helsinki_nlp_model.translate(sentence))

    nllb_model = NLLBModel()
    print(nllb_model.translate(sentence))

    t5_model = T5Model()
    print(t5_model.translate(sentence))


if __name__ == "__main__":
    main()
