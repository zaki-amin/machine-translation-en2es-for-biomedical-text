import json
import re

synonyms_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/preferred_synonyms_es.jsonl"


def build_preferred_synonyms_dictionary(filename: str) -> dict[str, str]:
    """Builds a preferred synonym dictionary, mapping from a term to its preferred synonym
    :param filename: the filename of the synonym dictionary in .jsonl
    :return: a dictionary from a term to its preferred term"""
    synonym_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            entry = json.loads(line)
            original, primary = entry["sec"], entry["ppal"]
            synonym_dict[original] = primary
    return synonym_dict


def postprocess_translation(phrase: str, synonyms_dictionary: dict[str, str]) -> str:
    for original in synonyms_dictionary.keys():
        if original in phrase:
            print(original)
            phrase = re.sub(original, synonyms_dictionary[original], phrase)
    return phrase


def main():
    synonyms_dictionary = build_preferred_synonyms_dictionary(synonyms_filename)
    phrase = 'El paciente parece sufrir problemas con el válvula del corazón'
    print(postprocess_translation(phrase, synonyms_dictionary))


if __name__ == "__main__":
    main()
