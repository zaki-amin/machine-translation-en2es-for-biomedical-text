import unittest

from domain_adaptation.corpus import get_all_filepaths, load_corpus


class TestCorpus(unittest.TestCase):
    def test_all_filepaths_constructed_correctly(self):
        directory_path = "../corpus/train/"
        filepaths = get_all_filepaths(directory_path)

        self.assertIn("../corpus/train/khresmoi.jsonl", filepaths, "khresmoi filepath not constructed correctly")
        self.assertEqual(len(filepaths), 8, "not all filepaths found")

    def test_corpus_data_loads_correctly(self):
        filepath = "../corpus/train/medline.jsonl"
        data = load_corpus(filepath, 0.2, 42)

        first_train_example = data["train"][0]
        self.assertEqual(first_train_example["en"], "Shin splints.")
        self.assertEqual(first_train_example["es"], "Dolor en las espinillas.")

        first_validation_example = data["validation"][0]
        self.assertEqual(first_validation_example["en"], "Micrognathia.")
        self.assertEqual(first_validation_example["es"], "Micrognacia.")
