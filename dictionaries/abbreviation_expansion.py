import json

from evaluation.official.string_similarity import SimilarityMetric

abbreviation_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"
metric = SimilarityMetric.SEMANTIC_SIMILARITY


def build_abbreviations_dictionary(filename: str) -> tuple[dict[str, list[str], dict[str, list[str]]]]:
    """Iterates over the abbreviation file and builds an English and Spanish abbreviation dictionary.
    Each abbreviation dictionary maps shorthand letters to a list of possible expansions.
    :param filename: the path to the abbreviation file
    :return: a tuple of two dictionaries, the first mapping English abbreviations to expansions and the second for Spanish abbreviations"""
    english_abbrs, spanish_abbrs = {}, {}
    with open(filename, 'r') as file:
        for line in file:
            abbreviation = json.loads(line)
            letters, expansion, abbr_type = abbreviation['abreviatura'], abbreviation['termino'], abbreviation[
                'cat_txt']
            match abbr_type:
                case 'AbrevEs' | 'Simbolo' | 'Formula' | 'Erroneo':
                    # add to the list if it already exists
                    if spanish_abbrs[letters]:
                        spanish_abbrs[letters].append(expansion)
                    else:
                        spanish_abbrs[letters] = [expansion]

                case 'AbrevEn':
                    if english_abbrs[letters]:
                        english_abbrs[letters].append(expansion)
                    else:
                        english_abbrs[letters] = [expansion]

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
    if expansions is None:
        return acronym

    best_expansion = expansions[0]
    best_similarity = metric.evaluate(phrase, best_expansion)
    for expansion in expansions[1:]:
        similarity = metric.evaluate(phrase, expansion)
        if similarity > best_similarity:
            best_similarity = similarity
            best_expansion = expansion
    return best_expansion


def contains_abbreviation(phrase: str,
                          abbreviation_dictionaries: tuple[dict[str, list[str], dict[str, list[str]]]],
                          language: str) -> bool:
    abbr_dict = abbreviation_dictionaries[0 if language == "en" else 1]
    for word in phrase.split(" "):
        if word in abbr_dict:
            return True
    return False


def main():
    abbr_dictionaries = build_abbreviations_dictionary(abbreviation_filename)
    print(contains_abbreviation("ACERGSGSDF is cool", abbr_dictionaries, "en"))
    print(contains_abbreviation("Es un ACE muy pequeño", abbr_dictionaries, "es"))


if __name__ == "__main__":
    main()
