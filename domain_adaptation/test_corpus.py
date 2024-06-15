import unittest

from domain_adaptation.corpus import get_all_filepaths, load_corpus, load_all_corpora


class TestCorpus(unittest.TestCase):
    directory_path = "../corpora/train/"
    filepaths = get_all_filepaths(directory_path)
    corpora = load_all_corpora(directory_path, 0.2, 42)

    def test_all_filepaths_constructed_correctly(self):
        self.assertIn("../corpus/train/khresmoi-tr.jsonl", self.filepaths,
                      "khresmoi filepath not constructed correctly")
        self.assertEqual(len(self.filepaths), 8, "not all filepaths found")

    def test_corpus_data_loads_correctly(self):
        filepath = "../corpora/train/medline.jsonl"
        data = load_corpus(filepath, 0.2, 42)

        first_train_example = data["train"][0]
        self.assertEqual(first_train_example["en"], "Do not give your baby cow's milk until they are 1 year old.")
        self.assertEqual(first_train_example["es"], "No le dé leche de vaca a su bebé hasta la edad de 1 año.")

        first_validation_example = data["validation"][0]
        self.assertEqual(first_validation_example["en"], "Stay at a healthy body weight.")
        self.assertEqual(first_validation_example["es"], "Mantener un peso corporal saludable.")

    def test_all_corpora_loaded(self):
        english_snomed_example = "Clonidine hydrochloride 100 microgram oral tablet."
        self.assertTrue(english_snomed_example in self.corpora["train"]["en"] or english_snomed_example in
                        self.corpora["validation"]["en"], "snomed corpus not loaded")
        spanish_medline_example = "Profesional de la salud mental."
        self.assertTrue(spanish_medline_example in self.corpora["train"]["es"] or spanish_medline_example in
                        self.corpora["validation"]["es"], "medline corpus not loaded")
