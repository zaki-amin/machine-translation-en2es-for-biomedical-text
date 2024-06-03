import json

from evaluations.sentence_similarity import SimilarityMetric


class Abbreviations:
    def __init__(self, abbr_file: str, pre_exp: bool = True, post_exp: bool = True):
        self.abbreviation_filename = abbr_file
        self.pre_exp = pre_exp
        self.post_exp = post_exp
        self.abbreviation_dictionary_en, self.abbreviation_dictionary_es = self._build_abbreviations_dictionary()
        self.semantic_similarity = SimilarityMetric.SEMANTIC_SIMILARITY

    def _build_abbreviations_dictionary(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Iterates over the abbreviation file and builds English and Spanish abbreviation dictionaries. Each
        abbreviation dictionary maps shorthand letters to a list of possible expansions.
        :return: a tuple of two
        dictionaries, the first mapping English abbreviations to expansions and the second for Spanish abbreviations"""
        english_abbrs, spanish_abbrs = {}, {}
        with open(self.abbreviation_filename, 'r') as file:
            for line in file:
                abbreviation = json.loads(line)
                acronym, term, category = abbreviation['acronym'], abbreviation['term'], abbreviation[
                    'category']
                match category:
                    case 'AbrevEs' | 'Simbolo' | 'Formula':
                        if acronym in spanish_abbrs:
                            spanish_abbrs[acronym].append(term)
                        else:
                            spanish_abbrs[acronym] = [term]

                    case 'AbrevEn':
                        if acronym in english_abbrs:
                            english_abbrs[acronym].append(term)
                        else:
                            english_abbrs[acronym] = [term]

                    case 'Erroneo':
                        # Discard these erroneous abbreviations
                        continue

        return english_abbrs, spanish_abbrs

    def most_appropriate_expansion(self,
                                   acronym: str,
                                   phrase: str,
                                   lang: str,
                                   similarity_threshold: float = 0.8) -> str:
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
        best_similarity = self.semantic_similarity.evaluate(phrase, phrase.replace(acronym, expansions[0]))
        for expansion in expansions[1:]:
            similarity = self.semantic_similarity.evaluate(phrase, phrase.replace(acronym, expansion))
            if similarity > best_similarity:
                best_similarity = similarity
                best_expansion = expansion

        if best_similarity < similarity_threshold:
            # Not similar enough to phrase to be considered appropriate
            return acronym
        return best_expansion

    def expand_all_abbreviations(self, phrase: str, lang: str) -> str:
        """Expands all abbreviations in a phrase in the corresponding langauge
        :param phrase: the phrase with abbreviations
        :param lang: the language of the phrase
        :returns: the phrase with all abbreviations expanded"""
        dictionary = self.abbreviation_dictionary_en if lang == "en" else self.abbreviation_dictionary_es
        for word in phrase.split(" "):
            # Any punctuation attached must be removed before checking membership
            word = word.strip(".,;:!?()[]{}")
            if word in dictionary:
                replacement = self.most_appropriate_expansion(word, phrase, lang)
                phrase = phrase.replace(word, replacement)
        return phrase

    def expand_all_abbreviations_english(self, phrase: str) -> str:
        """Expands all abbreviations in an English phrase.
        :param phrase: the phrase with abbreviations
        :returns: the phrase with all abbreviations expanded"""
        return self.expand_all_abbreviations(phrase, "en")

    def expand_all_abbreviations_spanish(self, phrase: str) -> str:
        """Expands all abbreviations in a Spanish phrase..
        :param phrase: the phrase with abbreviations
        :returns: the phrase with all abbreviations expanded"""
        return self.expand_all_abbreviations(phrase, "es")

    def preprocess(self, english_inputs: list[str]) -> list[str]:
        """If flag pre_exp enabled, expand all abbreviations in all the English inputs provided.
        Otherwise, return the English inputs as they are"""
        if not self.pre_exp:
            return english_inputs
        return list(map(lambda line: self.expand_all_abbreviations_english(line), english_inputs))

    def postprocess(self, spanish_outputs: list[str]) -> list[str]:
        """If flag post_exp enabled, expand all abbreviations in all the Spanish outputs provided.
        Otherwise, return the Spanish outputs as they are"""
        if not self.post_exp:
            return spanish_outputs
        return list(map(lambda line: self.expand_all_abbreviations_spanish(line), spanish_outputs))
