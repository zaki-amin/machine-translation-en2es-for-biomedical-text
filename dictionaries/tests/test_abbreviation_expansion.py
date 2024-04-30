import unittest

from dictionaries.abbreviations import Abbreviations


class TestAbbreviations(unittest.TestCase):
    abbreviation_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"
    abbr = Abbreviations(abbreviation_filename)

    def test_abbreviation_dict_english(self):
        self.assertTrue('AMP' in self.abbr.abbreviation_dictionary_en,
                        "AMP should be in the English abbreviation dictionary")
        self.assertEqual(self.abbr.abbreviation_dictionary_en["PD-1"], ["programmed cell death protein 1"],
                         "PD-1 should map to programmed cell death protein 1 in the dictionary")


if __name__ == '__main__':
    unittest.main()
