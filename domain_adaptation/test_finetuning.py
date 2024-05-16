import unittest

from domain_adaptation.finetuning import FineTuning, load_corpus, get_all_filepaths


class TestFineTuning(unittest.TestCase):
    fine_tuning = FineTuning("Helsinki-NLP/opus-mt-en-es", 512)
    filepath = "../corpus/train/medline.jsonl"
    data = load_corpus(filepath, 0.2, 42)

    def test_tokenizer_transforms_text(self):
        text = ("Underterminate colitis designates a rare inflammatory bowel disease that clinically resembles Crohn's "
                "disease and ulcerative colitis (see these terms) but that cannot be diagnosed as one of them after "
                "examination of an intestinal resection specimen.")
        tokens = self.fine_tuning.tokenizer(text)
        self.assertIn("input_ids", tokens, "Tokenization failed to produce input_ids")

    def test_both_training_and_validation_data_preprocessed(self):
        preprocessed_data = self.fine_tuning.tokenize_all_datasets(self.data)

        first_train_example = preprocessed_data["train"][0]
        self.assertIn("input_ids", first_train_example, "Training data not preprocessed")

        first_validation_example = preprocessed_data["validation"][0]
        self.assertIn("input_ids", first_validation_example, "Validation data not preprocessed")

    def test_data_collation(self):
        preprocessed_data = self.fine_tuning.tokenize_all_datasets(self.data)
        batch = self.fine_tuning.data_collator([preprocessed_data["train"][i] for i in range(1, 3)])
        self.assertIn("input_ids", batch, "Data collation failed to produce input_ids")
        self.assertIn("labels", batch, "Data collation failed to produce labels")

    def finetune_model(self):
        preprocessed_data = self.fine_tuning.tokenize_all_datasets(self.data)
        self.fine_tuning.finetune_model(self.data, preprocessed_data)
