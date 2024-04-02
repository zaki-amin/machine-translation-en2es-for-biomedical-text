import pandas as pd


def write_jsonl(filename: str):
    """Reads the official translations in TSV format and writes pairs to a JSONL file
    :param filename: path to the TSV file containing official translations
    """
    print("---Reading official translations---")
    df = pd.read_table(filename, delimiter='\t')
    print("---Writing to JSONL---")
    with open("official_translations.jsonl", "w") as file:
        for _, row in df.iterrows():
            file.write('{"en": "' + row["source_value"] + '", "es": "' + row["translation_value"] + '"}\n')


if __name__ == "__main__":
    write_jsonl("/Users/zaki/Desktop/Estudios/Master's thesis/Resources/HPO data/hp-es.babelon.tsv")
