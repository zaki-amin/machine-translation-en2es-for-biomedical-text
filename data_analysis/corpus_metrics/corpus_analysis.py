import json
import os
import string

import pandas as pd
from nltk.tokenize import word_tokenize


# Download the first time if necessary
# nltk.download('punkt')


def analyse_corpus(directory: str):
    """Analyses properties of all corpora in a directory and saves results to a .CSV file"""
    folder_path = f"../../corpus/{directory}"

    for root, dirs, files in os.walk(folder_path):
        results_df = pd.DataFrame(
            columns=['name', 'pairs', 'avg source length', 'avg target length', 'source vocabulary size',
                     'target vocabulary size', 'source punctuation rate', 'target punctuation rate'])

        for file_name in files:
            corpus_df = pd.DataFrame(columns=['en', 'es'])
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    en, es = data['en'], data['es']
                    corpus_df.loc[len(corpus_df.index)] = [en, es]

            results_df.loc[len(results_df.index)] = [file_name,
                                                     num_pairs(corpus_df),
                                                     avg_source_length(corpus_df),
                                                     avg_target_length(corpus_df),
                                                     source_vocabulary_size(corpus_df),
                                                     target_vocabulary_size(corpus_df),
                                                     punctuation_rate(corpus_df, 'en'),
                                                     punctuation_rate(corpus_df, 'es')]

        results_df.to_csv(f"{directory}.csv", index=False)


def num_pairs(df: pd.DataFrame) -> int:
    return df.shape[0]


def avg_source_length(df: pd.DataFrame) -> float:
    return df['en'].apply(lambda x: len(x.split())).mean()


def avg_target_length(df: pd.DataFrame) -> float:
    return df['es'].apply(lambda x: len(x.split())).mean()


def source_vocabulary_size(df: pd.DataFrame) -> int:
    return len(set(' '.join(df['en']).split()))


def target_vocabulary_size(df: pd.DataFrame) -> int:
    return len(set(' '.join(df['es']).split()))


def punctuation_rate(df: pd.DataFrame, lang) -> float:
    total_tokens = 0
    punctuation_tokens = 0
    for phrase in df[lang]:
        if lang == 'es':
            tokens = word_tokenize(phrase, language='spanish')
        else:
            tokens = word_tokenize(phrase)

        total_tokens += len(tokens)
        for token in tokens:
            if token in string.punctuation:
                punctuation_tokens += 1

    if total_tokens == 0:
        return 0.0
    return punctuation_tokens / total_tokens


if __name__ == '__main__':
    analyse_corpus("train")
    analyse_corpus("test")
