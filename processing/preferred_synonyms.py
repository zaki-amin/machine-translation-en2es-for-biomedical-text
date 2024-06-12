import json
import re
from collections import defaultdict

from tqdm import tqdm

from evaluations.sentence_similarity import SentenceSimilarity


class PreferredSynonyms:
    def __init__(self, synonyms_filename: str):
        self.synonyms_filename = synonyms_filename
        self.synonyms_dictionary = self._build_preferred_synonyms_dictionary()
        # Order synonym dictionary by length so longer terms are replaced first
        self.sorted_synonyms = sorted(self.synonyms_dictionary.items(), key=lambda x: len(x[0].split()), reverse=True)
        self.semantic_similarity = SentenceSimilarity.SEMANTIC_SIMILARITY

    def _build_preferred_synonyms_dictionary(self) -> dict[str, list[str]]:
        """Builds a preferred synonym dictionary, mapping from a term to its preferred synonym
        :return: a dictionary from a term to its preferred term"""
        synonym_dict = defaultdict(list)
        with open(self.synonyms_filename, 'r') as file:
            for line in file:
                entry = json.loads(line)
                original, primary = entry["secundario"], entry["principal"]
                synonym_dict[original].append(primary)
        return synonym_dict

    def postprocess(self, spanish: list[str]) -> list[str]:
        """Post-processes the Spanish translations, replacing terms with their preferred synonyms.
        :param spanish: the list of Spanish translations
        :return: the post-processed Spanish translations"""
        print("\nPost-processing translations with synonym replacement...")
        return [self._postprocess_translation(phrase) for phrase in tqdm(spanish)]

    def _postprocess_translation(self, phrase: str) -> str:
        def is_phrase_contained(candidate):
            # Only match entire words
            pattern = r'\b{}\b'.format(re.escape(candidate))
            return bool(re.search(pattern, phrase))

        for secondary, preferred in self.sorted_synonyms:
            if is_phrase_contained(secondary):
                best_replacement = self._best_replacement(phrase, secondary, preferred)
                phrase = phrase.replace(secondary, best_replacement)

        return phrase

    def _best_replacement(self, phrase: str, secondary: str, preferred: list[str]) -> str:
        if len(preferred) == 1:
            return preferred[0]

        best_replacement = preferred[0]
        best_similarity = self.semantic_similarity.evaluate(phrase, phrase.replace(secondary, best_replacement))
        for replacement in preferred[1:]:
            similarity = self.semantic_similarity.evaluate(phrase,  phrase.replace(secondary, replacement))
            if similarity > best_similarity:
                best_replacement = replacement
                best_similarity = similarity
        return best_replacement
