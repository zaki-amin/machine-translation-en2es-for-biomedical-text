import json
from collections import defaultdict


class PreferredSynonyms:
    def __init__(self, synonyms_filename: str):
        self.synonyms_filename = synonyms_filename
        self.synonyms_dictionary = self._build_preferred_synonyms_dictionary()

    def _build_preferred_synonyms_dictionary(self) -> dict[str, list[str]]:
        """Builds a preferred synonym dictionary, mapping from a term to its preferred synonym
        :return: a dictionary from a term to its preferred term"""
        synonym_dict = defaultdict(list)
        with open(self.synonyms_filename, 'r') as file:
            for line in file:
                entry = json.loads(line)
                original, primary = entry["sec"], entry["ppal"]
                synonym_dict[original].append(primary)
        return synonym_dict

    def postprocess_translation(self, phrase: str) -> str:
        return phrase
