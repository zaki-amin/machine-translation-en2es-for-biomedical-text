import unittest

from processing.abbreviations import Abbreviations


class TestAbbreviations(unittest.TestCase):
    abbreviation_filename = "../dictionaries/processed/abbreviations.jsonl"
    abbr = Abbreviations(abbreviation_filename, pre_exp=True, post_exp=True)

    def test_loading_abbreviation_dict_english(self):
        self.assertTrue('AMP' in self.abbr.abbreviation_dictionary_en,
                        "AMP should be in the English abbreviation dictionary")
        self.assertEqual(self.abbr.abbreviation_dictionary_en["PD-1"], ["programmed cell death protein 1"],
                         "PD-1 should map to programmed cell death protein 1 in the dictionary")

    def test_multiple_english_abbreviations_for_same_shorthand(self):
        adc_expansions = self.abbr.abbreviation_dictionary_en["ADC"]
        self.assertTrue("antibody-drug conjugate" in adc_expansions, "ADC should map to antibody-drug conjugate")
        self.assertTrue("drug-conjugated antibody" in adc_expansions, "ADC should map to drug-conjugated antibody")
        self.assertTrue("drug immunoconjugate" in adc_expansions, "ADC should map to drug immunoconjugate")

    def test_loading_abbreviation_dict_spanish(self):
        self.assertTrue('V/Q' in self.abbr.abbreviation_dictionary_es,
                        "V/Q should be in the Spanish abbreviation dictionary")
        self.assertEqual(self.abbr.abbreviation_dictionary_es["V/Q"], ["relación ventilación-perfusión"],
                         "V/Q should map to relación ventilación-perfusión in the dictionary")

    def test_multiple_spanish_abbreviations_for_same_shorthand(self):
        ai_expansions = self.abbr.abbreviation_dictionary_es["AI"]
        self.assertTrue("atrio izquierdo" in ai_expansions, "AI should map to atrio izquierdo")
        self.assertTrue("autoinmunitaria" in ai_expansions, "AI should map to autoinmunitaria")
        self.assertTrue("autoinmunitario" in ai_expansions, "AI should map to autoinmunitario")

    def test_discard_abbreviations_of_category_erroneo(self):
        self.assertFalse("hr" in self.abbr.abbreviation_dictionary_es,
                         "'hr' should not be an abbreviation for 'hora' in the Spanish abbreviation dictionary")

    def test_no_self_expansion(self):
        gds_expansions = self.abbr.abbreviation_dictionary_en["GDS"]
        self.assertFalse("GDS" in gds_expansions, "GDS should not map to GDS in the dictionary")

    def test_most_appropriate_expansion_english(self):
        phrase = "The test revealed there was AMP in his urine, a sign of prostate cancer"
        self.assertEqual(self.abbr.most_appropriate_expansion("AMP", phrase, 'en'), "adenosine monophosphate",
                         "AMP should expand to adenosine monophosphate in this context and not Academia de Medicina "
                         "del Paraguay")

    def test_whole_phrase_expansion_english(self):
        phrase = "The elderly man scored highly on the GDS, indicating depression"
        replacement_phrase = self.abbr.expand_all_abbreviations_english(phrase)
        self.assertEqual(replacement_phrase,
                         "The elderly man scored highly on the geriatric depression scale, indicating depression",
                         "GDS should expand to geriatric depression scale in this context")

    def test_most_appropriate_expansion_spanish(self):
        phrase = "El DM se manifiesta como piel seca, poliuria y polidipsia"
        self.assertEqual(self.abbr.most_appropriate_expansion("DM", phrase, 'es'), "dermatomiositis",
                         "DM should expand to dermatomiositis in this context and not diabetes mellitus")

    def test_whole_phrase_expansion_spanish(self):
        phrase = "El médico descubrió que el paciente tenía alta TA, aumentando el riesgo del ataque cardíaco"
        replacement_phrase = self.abbr.expand_all_abbreviations_spanish(phrase)
        self.assertEqual(replacement_phrase,
                         "El médico descubrió que el paciente tenía alta presión arterial, aumentando el riesgo "
                         "del ataque cardíaco",
                         "TA should expand to presión arterial in this context")

    def test_preprocessing_expands_abbreviations_attached_to_punctuation(self):
        phrase = "The clinic screens patients with the GDS: this is effective at finding loneliness in older adults."
        replacement_phrase = self.abbr.expand_all_abbreviations_english(phrase)
        self.assertEqual(replacement_phrase,
                         "The clinic screens patients with the geriatric depression scale: this is effective at "
                         "finding loneliness in older adults.")

    def test_expansion_is_case_sensitive(self):
        self.assertEqual(self.abbr.expand_all_abbreviations_spanish("si"),
                         "si",
                         "si should not expand to sistema internacional de unidades for SI")

    def test_preprocessing_expands_abbreviations_in_all_sentences(self):
        english_inputs = ["The patient's tRNA does not function properly, indicating ribosomal damage.",
                          "Her symptoms and travel history suggest she has contracted CHIKV."]
        preprocessed = self.abbr.preprocess(english_inputs)
        self.assertEqual(preprocessed[0],
                         "The patient's transfer RNA does not function properly, indicating ribosomal damage.",
                         "tRNA should expand to transfer RNA in this context")
        self.assertEqual(preprocessed[1],
                         "Her symptoms and travel history suggest she has contracted chikungunya virus.",
                         "CHIKV should expand to chikungunya virus in this context")

    def test_preprocessing_does_not_expand_abbreviations_when_flag_disabled(self):
        abbr = Abbreviations(self.abbreviation_filename, pre_exp=False, post_exp=True)
        english_inputs = ["The patient's tRNA does not function properly, indicating ribosomal damage.",
                          "Her symptoms and travel history suggest she has contracted CHIKV."]
        preprocessed = abbr.preprocess(english_inputs)
        self.assertEqual(preprocessed, english_inputs, "Abbreviations should not be expanded when pre_exp is False")

    def test_postprocessing_does_not_expand_abbreviations_when_flag_disabled(self):
        abbr = Abbreviations(self.abbreviation_filename, pre_exp=True, post_exp=False)
        spanish_outputs = [
            "El médico descubrió que el paciente tenía alta TA, aumentando el riesgo del ataque cardíaco"]
        postprocessed = abbr.postprocess(spanish_outputs)
        self.assertEqual(postprocessed, spanish_outputs, "Abbreviations should not be expanded when post_exp is False")


if __name__ == '__main__':
    unittest.main()
