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
        :param filename: the path to the abbreviation file
        :return: a tuple of two dictionaries, the first mapping English abbreviations to expansions and the second for Spanish abbreviations"""
        english_abbrs, spanish_abbrs = {}, {}
        with open(self.abbreviation_filename, 'r') as file:
            for line in file:
                abbreviation = json.loads(line)
                letters, expansion, abbr_type = abbreviation['abreviatura'], abbreviation['termino'], abbreviation[
                    'cat_txt']
                match abbr_type:
                    case 'AbrevEs' | 'Simbolo' | 'Formula' | 'Erroneo':
                        if letters in spanish_abbrs:
                            spanish_abbrs[letters].append(expansion)
                        else:
                            spanish_abbrs[letters] = [expansion]

                    case 'AbrevEn':
                        if letters in english_abbrs:
                            english_abbrs[letters].append(expansion)
                        else:
                            english_abbrs[letters] = [expansion]

        return english_abbrs, spanish_abbrs

    def most_appropriate_expansion(self,
                                   acronym: str,
                                   phrase: str,
                                   lang: str) -> str:
        """Chooses the most appropriate expansion for an acronym.
        No possible expansion returns the acronym as is
        Otherwise, returns the most appropriate expansion based on semantic similarity in the phrase context"""
        abbreviation_dictionary = self.abbreviation_dictionary_en if lang == 'en' else self.abbreviation_dictionary_es
        expansions = abbreviation_dictionary[acronym]
        if expansions is None:
            return acronym

        best_expansion = expansions[0]
        best_similarity = self.semantic_similarity.evaluate(phrase, best_expansion)
        for expansion in expansions[1:]:
            similarity = self.semantic_similarity.evaluate(phrase, expansion)
            if similarity > best_similarity:
                best_similarity = similarity
                best_expansion = expansion
        return best_expansion

    def expand_all_abbreviations_english(self, phrase: str) -> str:
        """Expands all abbreviations in an English phrase.
        :param phrase: the phrase to expand
        :returns: the phrase with all abbreviations expanded"""
        for word in phrase.split(" "):
            if word in self.abbreviation_dictionary_en:
                acronym = word
                replacement = self.most_appropriate_expansion(acronym, phrase, 'en')
                phrase = phrase.replace(acronym, replacement)
        return phrase


def main():
    abbreviation_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"
    abbr = Abbreviations(abbreviation_filename)

    phrase = "The test revealed there was AMP in his urine, a sign of prostate cancer"
    print(abbr.expand_all_abbreviations_english(phrase))


if __name__ == "__main__":
    main()
