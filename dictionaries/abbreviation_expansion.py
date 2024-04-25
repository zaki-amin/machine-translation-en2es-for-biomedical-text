import json
import pprint
from collections import defaultdict

abbreviation_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"


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


def main():
    abbreviation_dictionary = build_abbreviations_dictionary(abbreviation_filename)
    pprint.pprint(abbreviation_dictionary)


if __name__ == "__main__":
    main()
