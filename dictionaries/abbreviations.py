import json

from evaluation.official.string_similarity import SimilarityMetric


class Abbreviations:
    def __init__(self, abbr_file: str):
        self.abbreviation_filename = abbr_file
        self.abbreviation_dictionary_en, self.abbreviation_dictionary_es = self._build_abbreviations_dictionary()
        self.semantic_similarity = SimilarityMetric.SEMANTIC_SIMILARITY

    def _build_abbreviations_dictionary(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Iterates over the abbreviation file and builds an English and Spanish abbreviation dictionary.
        Each abbreviation dictionary maps shorthand letters to a list of possible expansions.
        :return: a tuple of two dictionaries, the first mapping English abbreviations to expansions
        and the second for Spanish abbreviations"""
        english_abbrs, spanish_abbrs = {}, {}
        with open(self.abbreviation_filename, 'r') as file:
            for line in file:
                abbreviation = json.loads(line)
                acronym, term, category = abbreviation['acronym'], abbreviation['term'], abbreviation[
                    'category']
                if acronym == term:
                    # Do not add acronyms which are mapped to themselves
                    continue
                match category:
                    case 'AbrevEs' | 'Simbolo' | 'Formula' | 'Erroneo':
                        if acronym in spanish_abbrs:
                            spanish_abbrs[acronym].append(term)
                        else:
                            spanish_abbrs[acronym] = [term]

                    case 'AbrevEn':
                        if acronym in english_abbrs:
                            english_abbrs[acronym].append(term)
                        else:
                            english_abbrs[acronym] = [term]

        return english_abbrs, spanish_abbrs

    def most_appropriate_expansion(self,
                                   acronym: str,
                                   phrase: str,
                                   lang: str,
                                   similarity_threshold: float = 0.4) -> str:
        """Chooses the most appropriate expansion for an acronym.
        No possible expansion returns the acronym as is
        Otherwise, returns the most appropriate expansion based on semantic similarity in the phrase context
        :param acronym: the acronym to expand
        :param phrase: the phrase containing the acronym
        :param lang: the language of the acronym
        :param similarity_threshold: the threshold above which an expansion is considered appropriate
        :returns: the most appropriate expansion for the acronym in the phrase context"""
        abbreviation_dictionary = self.abbreviation_dictionary_en if lang == 'en' else self.abbreviation_dictionary_es
        if acronym not in abbreviation_dictionary:
            return acronym

        expansions = abbreviation_dictionary[acronym]
        if len(expansions) == 1:
            # Choose the only possible expansion
            return expansions[0]

        best_expansion = expansions[0]
        best_similarity = self.semantic_similarity.evaluate(phrase, best_expansion)
        for expansion in expansions[1:]:
            similarity = self.semantic_similarity.evaluate(phrase, expansion)
            if similarity > best_similarity:
                best_similarity = similarity
                best_expansion = expansion

        if best_similarity < similarity_threshold:
            # Not similar enough to phrase to be considered appropriate
            return acronym
        return best_expansion

    def expand_all_abbreviations_english(self, phrase: str) -> str:
        """Expands all abbreviations in an English phrase. Pre-processing step.
        :param phrase: the phrase with abbreviations
        :returns: the phrase with all abbreviations expanded"""
        for word in phrase.split(" "):
            # Any punctuation attached must be removed before checking membership
            word = word.strip(".,;:!?()[]{}")
            if word in self.abbreviation_dictionary_en:
                replacement = self.most_appropriate_expansion(word, phrase, "en")
                phrase = phrase.replace(word, replacement)
        return phrase

    def expand_all_abbreviations_spanish(self, phrase: str) -> str:
        """Expands all abbreviations in a Spanish phrase. Post-processing step.
        Maintains the initial acronym and puts its meaning in brackets after.
        :param phrase: the phrase with abbreviations
        :returns: the phrase with all abbreviations expanded"""
        for word in phrase.split(" "):
            # Any punctuation attached must be removed before checking membership
            word = word.strip(".,;:!?()[]{}")
            if word in self.abbreviation_dictionary_es:
                replacement = self.most_appropriate_expansion(word, phrase, "es")
                phrase = phrase.replace(word, replacement)
        return phrase

    def preprocess(self, english_inputs: list[str]) -> list[str]:
        """Expands all abbreviations in all of the English inputs provided"""
        return list(map(lambda line: self.expand_all_abbreviations_english(line), english_inputs))
