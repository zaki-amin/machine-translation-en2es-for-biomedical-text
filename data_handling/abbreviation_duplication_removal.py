import json


def remove_duplicate_abbreviations(input_file: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as out:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                acronym, term, category = entry["acronym"], entry["term"], entry["category"]
                if acronym != term:
                    out.write(json.dumps({"acronym": acronym, "term": term, "category": category}) + "\n")


if __name__ == '__main__':
    remove_duplicate_abbreviations(
        "/Users/zaki/PycharmProjects/hpo_translation/processing/dictionaries/processed/abbreviations.jsonl",
        "abbreviations.jsonl")
