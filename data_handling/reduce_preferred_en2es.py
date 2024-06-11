import json

jsonl_file = "../processing/dictionaries/processed/preferred_en2es.jsonl"


def reduce_preferred_en2es(input_file: str, output_file: str):
    with open(output_file, 'w') as out:
        with open(input_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                en, es = entry["en"], entry["es"]
                out.write(json.dumps({"en": en, "es": es}) + "\n")


if __name__ == '__main__':
    reduce_preferred_en2es(jsonl_file, "reduced.jsonl")
