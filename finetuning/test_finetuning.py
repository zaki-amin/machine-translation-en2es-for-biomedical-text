import unittest

from finetuning.data import load_corpus


class TestFineTuning(unittest.TestCase):
    def test_data_loads_correctly(self):
        filepath = '../corpus/train/medline.jsonl'
        data = load_corpus(filepath, 0.2, 42)

        first_train_example = data['train'][0]
        self.assertEqual(first_train_example['en'], 'Shin splints.')
        self.assertEqual(first_train_example['es'], 'Dolor en las espinillas.')

        first_validation_example = data['validation'][0]
        self.assertEqual(first_validation_example['en'], 'Micrognathia.')
        self.assertEqual(first_validation_example['es'], 'Micrognacia.')
