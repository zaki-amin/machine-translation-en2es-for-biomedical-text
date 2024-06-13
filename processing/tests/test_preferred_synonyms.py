import unittest

from processing.preferred_synonyms import PreferredSynonyms


class TestPreferredSynonyms(unittest.TestCase):
    synonyms_filename = "../dictionaries/processed/preferred-synonyms-es.jsonl"
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

    def test_simple_replacement(self):
        phrase = 'La médica recomienda el uso de un método contraceptivo translations el parto durante tres meses'
        self.assertEqual(self.preferred_synonyms._postprocess_translation(phrase),
                         'La médica recomienda el uso de un método anticonceptivo translations el parto durante tres meses',
                         "synonym replacement should change 'método contraceptivo' to 'método anticonceptivo'")

    def test_can_chain_synonym_replacement(self):
        # Should replace 'válvula del corazón' to 'válvula cardíaca' to 'prótesis valvular'
        phrase = 'El paciente parece sufrir problemas con la válvula del corazón'
        self.assertEqual('El paciente parece sufrir problemas con la prótesis valvular',
                         self.preferred_synonyms._postprocess_translation(phrase),
                         "double synonym replacement expected")

    def test_postprocess_multiple_translations(self):
        original = ["No se recomienda entrar en un hospital mental en casos leves de la depresión.",
                    "La fibra amarilla juega un papel crucial en la estructura de las arterias."]
        postprocessed = self.preferred_synonyms.postprocess(original)

        expected = ["No se recomienda entrar en un hospital psiquiátrico en casos leves de la depresión.",
                    "La fibra elástica juega un papel crucial en la estructura de las arterias."]
        self.assertEqual(postprocessed[0], expected[0])
        self.assertEqual(postprocessed[1], expected[1])

    def test_postprocessing_works_with_punctuation_attached(self):
        phrase = "La vena poscava: el principal vaso sanguíneo que transporta la sangre desoxigenada"
        self.assertEqual(self.preferred_synonyms._postprocess_translation(phrase),
                         "La vena cava inferior: el principal vaso sanguíneo que transporta la sangre desoxigenada",
                         "synonym replacement should change 'vena poscava' to 'vena cava posterior'")

    def test_chooses_longest_match(self):
        phrase = "Se aprobó el virus de la hepatitis infecciosa en los años 90"
        self.assertEqual(self.preferred_synonyms._postprocess_translation(phrase),
                         "Se aprobó el virus de la hepatitis A en los años 90")

    def test_choose_longest_match2(self):
        phrase = ("El tronco arterial pulmonar es una arteria que lleva sangre desoxigenada desde el corazón a los "
                  "pulmones")
        self.assertEqual(self.preferred_synonyms._postprocess_translation(phrase),
                         "El tronco pulmonar es una arteria que lleva sangre desoxigenada desde el corazón a los "
                         "pulmones",
                         'Replace "tronco arterial pulmonar to tronco pulmonar" instead of "tronco arterial" to '
                         '"tronco arterioso"')
