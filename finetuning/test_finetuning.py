import unittest

from finetuning.data import load_corpus, marian_tokenizer, tokenize_all_datasets


class TestFineTuning(unittest.TestCase):
    filepath = '../corpus/train/medline.jsonl'
    data = load_corpus(filepath, 0.2, 42)
    tokenizer = marian_tokenizer()

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
        tokens = self.tokenizer(text)
        self.assertIn("input_ids", tokens, "Tokenization failed to produce input_ids")
        print(tokens)

    def test_both_training_and_validation_data_preprocessed(self):
        max_length = 512
        preprocessed_data = tokenize_all_datasets(self.tokenizer, self.data, max_length)

        first_train_example = preprocessed_data['train'][0]
        self.assertIn("input_ids", first_train_example, "Training data not preprocessed")

        first_validation_example = preprocessed_data['validation'][0]
        self.assertIn("input_ids", first_validation_example, "Validation data not preprocessed")
