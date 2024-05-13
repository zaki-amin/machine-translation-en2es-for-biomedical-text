import json


def create_jsonl_files(en_filename: str, es_filename: str, term_output: str, definition_output: str):
    """Reads in equivalent Orphanet clinical terms from two files.
    Writes two .JSONL files, one with paired terms and one with paired definitions"""
    en_terms = read_terms_from_json(en_filename)
    es_terms = read_terms_from_json(es_filename)
    write_lists_to_jsonl(en_terms, es_terms, term_output)

    en_definitions, es_definitions, missing = read_definitions_from_jsons(en_filename, es_filename)
    print(f"Missing definitions: {missing}")
    write_lists_to_jsonl(en_definitions, es_definitions, definition_output)


def read_terms_from_json(filename: str) -> list[str]:
    """Reads a list of terms from a JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
        return [item['Preferred term'] for item in data]


def read_definitions_from_jsons(filename1: str, filename2: str) -> tuple[list[str], list[str], int]:
    """Reads definitions from the two .JSONL files provided.
    It is possible for definitions to be missing so count and exclude these"""
    with open(filename1, 'r') as f:
        data1 = json.load(f)
    with open(filename2, 'r') as f:
        data2 = json.load(f)
    definitions1 = [item.get('Definition') for item in data1]
    # Some Spanish definition keys are missing entirely
    definitions2 = [item.get('Definition') for item in data2]

    kept_definitions1, kept_definitions2 = [], []
    missing_definitions = 0
    for def1, def2 in zip(definitions1, definitions2):
        if def1 is None or def1 == "None available" or def2 is None or def2 == "None available":
            missing_definitions += 1
        else:
            kept_definitions1.append(def1)
            kept_definitions2.append(def2)

    return kept_definitions1, kept_definitions2, missing_definitions


def write_lists_to_jsonl(english: list[str], spanish: list[str], output_file: str):
    """Writes two lists into a .JSONL file with one pair per line"""
    with open(output_file, 'w') as f:
        for en, es in zip(english, spanish):
            json.dump({'en': en, 'es': es}, f)
            f.write('\n')


def main():
    en_filename = "clin_en.json"
    es_filename = "clin_es.json"
    create_jsonl_files(en_filename, es_filename, "orphanet_terms.jsonl", "orphanet_definitions.jsonl")


if __name__ == '__main__':
    main()
