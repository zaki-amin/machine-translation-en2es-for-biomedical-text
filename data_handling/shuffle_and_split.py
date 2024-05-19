import json

import numpy as np
from sklearn.model_selection import train_test_split


def shuffle_and_split(input_jsonl: str,
                      test_proportion: float,
                      seed: int,
                      output_name: str):
    """Reads in data, shuffles it and splits into two files"""
    english, spanish = [], []
    with open(input_jsonl, 'r') as file:
        for line in file:
            data = json.loads(line)
            en, es = data["en"], data["es"]
            english.append(en)
            spanish.append(es)

    english = np.array(english)
    spanish = np.array(spanish)
    en_train, en_test, es_train, es_test = train_test_split(english,
                                                            spanish,
                                                            test_size=test_proportion,
                                                            random_state=seed)
    with open(f"{output_name}-tr.jsonl", 'w') as train:
        for en, es in zip(en_train, es_train):
            train.write(json.dumps({"en": en, "es": es}) + "\n")
    with open(f"{output_name}-te.jsonl", 'w') as test:
        for en, es in zip(en_test, es_test):
            test.write(json.dumps({"en": en, "es": es}) + "\n")


if __name__ == '__main__':
    input_filename = "orphanet_definitions.jsonl"
    shuffle_and_split(input_filename, 0.1, 64, "orphanet-definitions")
