import json
import re
from collections import defaultdict

from hpo.official.string_similarity import SimilarityMetric


class PreferredSynonyms:
    def __init__(self, synonyms_filename: str):
        self.synonyms_filename = synonyms_filename
        self.synonyms_dictionary = self._build_preferred_synonyms_dictionary()
        self.semantic_similarity = SimilarityMetric.SEMANTIC_SIMILARITY

    def _build_preferred_synonyms_dictionary(self) -> dict[str, list[str]]:
        """Builds a preferred synonym dictionary, mapping from a term to its preferred synonym
        :return: a dictionary from a term to its preferred term"""
        synonym_dict = defaultdict(list)
        with open(self.synonyms_filename, 'r') as file:
            for line in file:
                entry = json.loads(line)
                original, primary = entry["sec"], entry["ppal"]
                num_words = len(original.split())
                if num_words >= 2:
                    # Avoid overreplacement by discarding single words
                    synonym_dict[original].append(primary)
        return synonym_dict

    def postprocess_translation(self, phrase: str) -> str:
        def is_phrase_contained(candidate):
            # Only match standalone words
            pattern = r'\b{}\b'.format(re.escape(candidate))
            return bool(re.search(pattern, phrase))

        for original, preferred in self.synonyms_dictionary.items():
            if is_phrase_contained(original):
                best_replacement = self._best_replacement(phrase, preferred)
                print(f"{original} -> {best_replacement}")
                phrase = phrase.replace(original, best_replacement)

        return phrase

    def _best_replacement(self, phrase: str, preferred: list[str]) -> str:
        if len(preferred) == 1:
            return preferred[0]

        best_replacement = preferred[0]
        best_similarity = self.semantic_similarity.evaluate(phrase, best_replacement)
        for replacement in preferred[1:]:
            similarity = self.semantic_similarity.evaluate(phrase, replacement)
            if similarity > best_similarity:
                best_replacement = replacement
                best_similarity = similarity
        return best_replacement
