import pandas as pd

from dictionaries.abbreviations import Abbreviations
from evaluation.official.string_similarity import SimilarityMetric


def translate_input(input_file: str):
    """Translates the input .txt file from English to Spanish.
    Applies pre-processing, the model and post-processing.
    Writes the output CSV to a file with the following columns:
    - 'original': the original English sentence
    - 'translation': the actual provided translation

    :param input_file: the input file to translate, a text file with each sentence on a new line"""
    with open(input_file, 'r') as file:
        english_inputs = [line.strip() for line in file.readlines()]
    english_inputs = preprocess(english_inputs)
    # do some model stuff here
    pass


def preprocess(english_inputs: list[str], abbreviations_filename: str) -> list[str]:
    abbreviations = Abbreviations(abbreviations_filename)
    return list(map(lambda line: abbreviations.expand_all_abbreviations_english(line), english_inputs))


def postprocess(df: pd.DataFrame, synonyms_filename: str) -> list[str]:
    pass


def evaluate_translations(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates the translations, adding metrics to the dataframe.
    The new columns are 'sacrebleu' for the SacreBLEU score, 'semsim' for the semantic similarity score
    and 'wer' for the word error rate."""

    def sacrebleu(row):
        return SimilarityMetric.SACREBLEU.evaluate(row['reference'], row['translation'])

    def semsim(row):
        return SimilarityMetric.SEMANTIC_SIMILARITY.evaluate(row['reference'], row['translation'])

    def wer(row):
        return SimilarityMetric.EDIT_DISTANCE.evaluate(row['reference'], row['translation'])

    df['sacrebleu'] = df.apply(sacrebleu, axis=1)
    df['semsim'] = df.apply(semsim, axis=1)
    df['wer'] = df.apply(wer, axis=1)
    return df


def translate_and_evaluate(input_file: str, output_file: str):
    """Translates the input .JSONL file from English to Spanish.
    Applies pre-processing, the model and post-processing.
    Evaluates the translations with various metrics.

    Writes the output CSV to a file with the following columns:
    - 'original': the original English sentence
    - 'reference': the actual provided translation
    - 'translation': the translation from the model
    - 'sacrebleu': the SacreBLEU score of the translation
    - 'semsim': the semantic similarity score of the translation

    :param input_file: the input file to translate and evaluate"""
    pass
