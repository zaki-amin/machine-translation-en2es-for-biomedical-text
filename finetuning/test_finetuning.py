import unittest

from finetuning.data import load_corpus


class TestFineTuning(unittest.TestCase):
    def test_data_loads_correctly(self):
        filepath = '../corpus/train/medline.jsonl'
        data = load_corpus(filepath)
        first_line = data[0]
        self.assertEqual(first_line['en'],
                         'To check for nystagmus, the provider may use the following procedure.')
        self.assertEqual(first_line['es'],
                         'Para examinar el nistagmo, el proveedor puede usar el siguiente procedimiento.')
