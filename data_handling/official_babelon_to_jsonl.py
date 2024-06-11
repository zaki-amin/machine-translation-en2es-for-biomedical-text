import json

import pandas as pd


def write_jsonl(filename: str):
    """Reads the official translations in TSV format and writes pairs to a JSONL file
    :param filename: path to the TSV file containing official translations
    """
    df = pd.read_table(filename, delimiter='\t')

    with open("official_translations.jsonl", "w") as jsonl_file:
        for _, row in df.iterrows():
            data = {"en": row["source_value"], "es": row["translation_value"]}
            jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    write_jsonl("../translation/hpo/official/hp-es.babelon.tsv")
