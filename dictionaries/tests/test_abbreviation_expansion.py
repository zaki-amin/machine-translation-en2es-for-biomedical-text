import unittest

from dictionaries.abbreviations import Abbreviations


class TestAbbreviations(unittest.TestCase):
    abbreviation_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"
    abbr = Abbreviations(abbreviation_filename)

    def test_loading_abbreviation_dict_english(self):
        self.assertTrue('AMP' in self.abbr.abbreviation_dictionary_en,
                        "AMP should be in the English abbreviation dictionary")
        self.assertEqual(self.abbr.abbreviation_dictionary_en["PD-1"], ["programmed cell death protein 1"],
                         "PD-1 should map to programmed cell death protein 1 in the dictionary")

    def test_multiple_english_abbreviations(self):
        adc_expansions = self.abbr.abbreviation_dictionary_en["ADC"]
        self.assertTrue("antibody-drug conjugate" in adc_expansions, "ADC should map to antibody-drug conjugate")
        self.assertTrue("drug-conjugated antibody" in adc_expansions, "ADC should map to drug-conjugated antibody")
        self.assertTrue("drug immunoconjugate" in adc_expansions, "ADC should map to drug immunoconjugate")

    def test_loading_abbreviation_dict_spanish(self):
        self.assertTrue('V/Q' in self.abbr.abbreviation_dictionary_es,
                        "V/Q should be in the Spanish abbreviation dictionary")
        self.assertEqual(self.abbr.abbreviation_dictionary_es["V/Q"], ["relación ventilación-perfusión"],
                         "V/Q should map to relación ventilación-perfusión in the dictionary")

    def test_multiple_spanish_abbreviations(self):
        ai_expansions = self.abbr.abbreviation_dictionary_es["AI"]
        self.assertTrue("atrio izquierdo" in ai_expansions, "AI should map to atrio izquierdo")
        self.assertTrue("autoinmunitaria" in ai_expansions, "AI should map to autoinmunitaria")
        self.assertTrue("autoinmunitario" in ai_expansions, "AI should map to autoinmunitario")

    def test_no_self_expansion(self):
        gds_expansions = self.abbr.abbreviation_dictionary_en["GDS"]
        self.assertFalse("GDS" in gds_expansions, "GDS should not map to GDS in the dictionary")

    def test_most_appropriate_expansion_english(self):
        phrase = "The test revealed there was AMP in his urine, a sign of prostate cancer"
        self.assertEqual(self.abbr.most_appropriate_expansion("AMP", phrase, 'en'), "adenosine monophosphate",
                         "AMP should expand to adenosine monophosphate in this context")

    def test_whole_phrase_expansion(self):
        phrase = "The elderly man scored highly on the GDS, indicating depression"
        replacement_phrase = self.abbr.expand_all_abbreviations_english(phrase)
        print(replacement_phrase)
        self.assertEqual(replacement_phrase,
                         "The elderly man scored highly on the geriatric depression scale, indicating depression",
                         "GDS should expand to geriatric depression scale in this context")


if __name__ == '__main__':
    unittest.main()
