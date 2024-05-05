import json

import pandas as pd

from dictionaries.abbreviations import Abbreviations
from dictionaries.preferred_synonyms import PreferredSynonyms
from evaluation.official.string_similarity import SimilarityMetric
from hpo_translator.src.models import MarianMTConfig, NMTModelConfig
from hpo_translator.src.translate import translate


def translate_input(input_file: str,
                    output_file: str,
                    abbreviations_filename: str,
                    synonyms_filename: str,
                    checkpoint: str | None = None,
                    config: NMTModelConfig = MarianMTConfig()):
    """Translates English from a .txt file to Spanish.
    Each line should have an English sentence.
    Applies pre-processing, the model and post-processing.
    Writes the output CSV to a file with the following columns:
    - 'en': the original English sentence
    - 'es': the model's translation

    :param input_file: the input file to translate, a .txt file with each sentence on a new line
    :param output_file: the output file to write results to
    :param abbreviations_filename: the filename of the abbreviations dictionary
    :param synonyms_filename: the filename of the synonyms dictionary
    :param checkpoint: the name of the model checkpoint to use. If not given, defaults to clinical MarianMT
    :param config: the configuration of the model to use. If not given, defaults to MarianMTConfig"""
    with open(input_file, 'r') as file:
        english_inputs = [line.strip() for line in file.readlines()]
    spanish_outputs = translate_english_inputs(english_inputs, abbreviations_filename, synonyms_filename, checkpoint,
                                               config)
    data = {'en': english_inputs, 'es': spanish_outputs}
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, header=True)


def translate_and_evaluate(input_file: str,
                           output_file: str,
                           abbreviations_filename: str,
                           synonyms_filename: str,
                           checkpoint: str | None = None,
                           config: NMTModelConfig = MarianMTConfig()):
    """Translates English and evaluates against reference translations from a .JSONL file.
    Each line should have an "en" and "es" key.
    Applies pre-processing, the model and post-processing.
    Evaluates the translations with various metrics.

    Writes the output CSV to a file with the following columns:
    - 'original': the original English sentence
    - 'reference': the actual provided translation
    - 'translation': the translation from the model
    - 'sacrebleu': the SacreBLEU score of the translation
    - 'wer': the word error rate of the translation
    - 'semsim': the semantic similarity score of the translation

    :param input_file: the input file to translate, a .txt file with each sentence on a new line
    :param output_file: the output file to write results to
    :param abbreviations_filename: the filename of the abbreviations dictionary
    :param synonyms_filename: the filename of the synonyms dictionary
    :param checkpoint: the name of the model checkpoint to use. If not given, defaults to clinical MarianMT
    :param config: the configuration of the model to use. If not given, defaults to MarianMTConfig"""
    english_texts, spanish_references = [], []
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            english_texts.append(entry['en'])
            spanish_references.append(entry['es'])

    spanish_outputs = translate_english_inputs(english_texts, abbreviations_filename, synonyms_filename, checkpoint,
                                               config)

    data = {'english': english_texts, 'reference': spanish_references, 'translation': spanish_outputs}
    df = pd.DataFrame(data)
    df = evaluate_translations(df)
    df.to_csv(output_file, index=False, header=True)


def translate_english_inputs(english_inputs: list[str],
                             abbreviations_filename: str,
                             synonyms_filename: str,
                             checkpoint: str | None,
                             config: NMTModelConfig) -> list[str]:
    """Translates English inputs to Spanish outputs. Applies pre-processing, the model and post-processing."""
    # Preprocessing
    abbreviations = Abbreviations(abbreviations_filename)
    english_inputs = abbreviations.preprocess(english_inputs)

    # Translation with model
    spanish_outputs = translate(english_inputs, checkpoint, config)

    # Postprocessing
    preferred_synonyms = PreferredSynonyms(synonyms_filename)
    spanish_outputs = abbreviations.postprocess(spanish_outputs)
    return spanish_outputs


def evaluate_translations(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates the translations, adding metrics to the dataframe.
    The new columns are 'sacrebleu' for the SacreBLEU score, 'semsim' for the semantic similarity score
    and 'wer' for the word error rate."""

    round_digits = 1

    def sacrebleu(row):
        score = SimilarityMetric.SACREBLEU.evaluate(row['reference'], row['translation'])
        return round(score * 100, round_digits)

    def semsim(row):
        score = SimilarityMetric.SEMANTIC_SIMILARITY.evaluate(row['reference'], row['translation'])
        return round(score * 100, round_digits)

    def wer(row):
        score = 1 - SimilarityMetric.WER.evaluate(row['reference'], row['translation'])
        return round(score * 100, round_digits)

    df['sacrebleu'] = df.apply(sacrebleu, axis=1)
    df['semsim'] = df.apply(semsim, axis=1)
    df['wer'] = df.apply(wer, axis=1)
    return df


def main():
    abbreviations_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl"
    synonyms_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/preferred_synonyms_es.jsonl"
    # translate_input("input.txt", "output.csv", abbreviations_filename, synonyms_filename)
    translate_and_evaluate("input.jsonl", "output.csv", abbreviations_filename, synonyms_filename)


if __name__ == "__main__":
    main()
