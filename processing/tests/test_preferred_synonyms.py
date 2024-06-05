import unittest

from processing.preferred_synonyms import PreferredSynonyms


class TestPreferredSynonyms(unittest.TestCase):
    synonyms_filename = ("/Users/zaki/PycharmProjects/hpo_translation/processing/dictionaries/processed"
                         "/preferred_synonyms_es.jsonl")
    preferred_synonyms = PreferredSynonyms(synonyms_filename)

    def test_synonym_dictionary_built_correctly(self):
        original = "absorciometría de fotón doble"
        self.assertEqual(self.preferred_synonyms.synonyms_dictionary[original], ["absorciometría fotónica dual"],
                         "preferred synonym dictionary should map to 'absorciometría fotónica dual'")

    def test_synonym_dictionary_ignores_one_word_synonyms(self):
        original = "ácido-base"
        self.assertNotIn("ácido-básico", self.preferred_synonyms.synonyms_dictionary[original],
                         "preferred synonym dictionary should not include masculine 'ácido-básico'")
        self.assertNotIn("ácido-básica", self.preferred_synonyms.synonyms_dictionary[original],
                         "preferred synonym dictionary should not include feminine 'ácido-básica'")

    def test_postprocess_translation1(self):
        phrase = 'El paciente parece sufrir problemas con la válvula del corazón'
        self.assertEqual(self.preferred_synonyms._postprocess_translation(phrase),
                         'El paciente parece sufrir problemas con la válvula cardíaca',
                         "synonym replacement should change 'válvula del corazón' to 'válvula cardíaca'")

    def test_postprocess_translation2(self):
        phrase = 'La médica recomienda el uso de un método contraceptivo tras el parto durante tres meses'
        self.assertEqual(self.preferred_synonyms._postprocess_translation(phrase),
                         'La médica recomienda el uso de un método anticonceptivo tras el parto durante tres meses',
                         "synonym replacement should change 'método contraceptivo' to 'método anticonceptivo'")
