import unittest

from domain_adaptation.finetuning import FineTuning


class TestFineTuning(unittest.TestCase):
    fine_tuning = FineTuning('Helsinki-NLP/opus-mt-en-es')
    filepath = '../corpus/train/medline.jsonl'
    data = fine_tuning.load_corpus(filepath, 0.2, 42)

    def test_data_loads_correctly(self):
        first_train_example = self.data['train'][0]
        self.assertEqual(first_train_example['en'], 'Shin splints.')
        self.assertEqual(first_train_example['es'], 'Dolor en las espinillas.')

        first_validation_example = self.data['validation'][0]
        self.assertEqual(first_validation_example['en'], 'Micrognathia.')
        self.assertEqual(first_validation_example['es'], 'Micrognacia.')

    def test_tokenizer_transforms_text(self):
        text = ("Underterminate colitis designates a rare inflammatory bowel disease that clinically resembles Crohn's "
                "disease and ulcerative colitis (see these terms) but that cannot be diagnosed as one of them after "
                "examination of an intestinal resection specimen.")
        tokens = self.fine_tuning.tokenizer(text)
        self.assertIn("input_ids", tokens, "Tokenization failed to produce input_ids")
        print(tokens)

    def test_both_training_and_validation_data_preprocessed(self):
        max_length = 512
        preprocessed_data = self.fine_tuning.tokenize_all_datasets(self.data, max_length)

        first_train_example = preprocessed_data['train'][0]
        self.assertIn("input_ids", first_train_example, "Training data not preprocessed")

        first_validation_example = preprocessed_data['validation'][0]
        self.assertIn("input_ids", first_validation_example, "Validation data not preprocessed")
