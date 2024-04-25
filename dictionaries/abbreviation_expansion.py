import json
from collections import defaultdict

from evaluation.official.string_similarity import SimilarityMetric

abbreviation_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"
metric = SimilarityMetric.SEMANTIC_SIMILARITY


def build_abbreviations_dictionary(filename: str) -> tuple[dict[str, list[str], dict[str, list[str]]]]:
    """Iterates over the abbreviation file and builds an English and Spanish abbreviation dictionary.
    Each abbreviation dictionary maps shorthand letters to a list of possible expansions.
    :param filename: the path to the abbreviation file
    :return: a tuple of two dictionaries, the first mapping English abbreviations to expansions and the second for Spanish abbreviations"""
    english_abbrs, spanish_abbrs = defaultdict(list[str]), defaultdict(list[str])
    with open(filename, 'r') as file:
        for line in file:
            abbreviation = json.loads(line)
            letters, expansion, abbr_type = abbreviation['abreviatura'], abbreviation['termino'], abbreviation[
                'cat_txt']
            match abbr_type:
                case 'AbrevEs' | 'Simbolo' | 'Formula' | 'Erroneo':
                    # add to the list if it already exists
                    spanish_abbrs[letters].append(expansion)
                case 'AbrevEn':
                    english_abbrs[letters].append(expansion)

    return english_abbrs, spanish_abbrs


def abbreviation_expansion(acronym: str,
                           phrase: str,
                           abbreviation_dictionaries: tuple[dict[str, list[str], dict[str, list[str]]]],
                           language: str) -> str:
    """Expands an acronym to the most appropriate word according to the language.
    :param acronym: the acronym to expand
    :param phrase: the entire sentence which contains the acronym for context in expansion
    :param abbreviation_dictionaries: the two abbreviation dictionaries for English and Spanish
    :param language: the language of the phrase, either 'en' or 'es'
    :returns: the best possible expanded acronym"""
    if language == "en":
        return most_appropriate_expansion(acronym, phrase, abbreviation_dictionaries[0])
    else:
        return most_appropriate_expansion(acronym, phrase, abbreviation_dictionaries[1])


def most_appropriate_expansion(acronym: str, phrase: str, abbreviation_dictionary: dict[str, list[str]]) -> str:
    """Chooses the most appropriate expansion for an acronym.
    No possible expansion returns the acronym as is
    Otherwise, returns the most appropriate expansion based on semantic similarity in the phrase context"""
    expansions = abbreviation_dictionary[acronym]
    num_expansions = len(expansions)
    if num_expansions == 0:
        return acronym
    elif num_expansions == 1:
        return expansions[0]

    best_expansion = expansions[0]
    best_similarity = metric.evaluate(phrase, best_expansion)
    for expansion in expansions[1:]:
        similarity = metric.evaluate(phrase, expansion)
        if similarity > best_similarity:
            best_similarity = similarity
            best_expansion = expansion
    return best_expansion


def main():
    abbr_dictionaries = build_abbreviations_dictionary(abbreviation_filename)
    phrase = "Antigens and ACE"
    acronym = "ACE"
    print(abbreviation_expansion(acronym, phrase, abbr_dictionaries, 'en'))
    print(abbreviation_expansion(acronym, phrase, abbr_dictionaries, 'es'))


if __name__ == "__main__":
    main()
