import json

jsonl_file = "../processing/dictionaries/processed/preferred_synonyms_es.jsonl"


def extract_more_than_one_word_synonyms(input_file: str, output_file: str):
    with open(output_file, 'w') as out:
        with open(input_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                secondary, primary = entry["sec"], entry["ppal"]
                if len(secondary.split()) > 1:
                    out.write(json.dumps({"secundario": secondary, "principal": primary}) + "\n")


if __name__ == '__main__':
    extract_more_than_one_word_synonyms(jsonl_file, "../processing/dictionaries/processed/reduced.jsonl")
