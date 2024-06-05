import argparse
import json

import pandas as pd

from evaluations.sentence_similarity import SimilarityMetric
from processing.abbreviations import Abbreviations
from processing.preferred_synonyms import PreferredSynonyms
from translation.translate import translate_text


def translate_no_evaluate(input_file: str,
                          output_file: str,
                          abbreviations_filename: str,
                          synonyms_filename: str,
                          expansions: tuple[bool, bool],
                          checkpoint: str):
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
    :param expansions: flags for pre- and post- abbreviation expansion
    :param checkpoint: the name of the model checkpoint to use."""
    with open(input_file, 'r') as file:
        english_inputs = [line.strip() for line in file.readlines()]
    spanish_outputs = translate_english_inputs(english_inputs,
                                               abbreviations_filename,
                                               synonyms_filename,
                                               expansions,
                                               checkpoint)
    data = {'english': english_inputs, 'translation': spanish_outputs}
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, header=True)


def translate_and_evaluate(input_file: str,
                           output_file: str,
                           abbreviations_filename: str,
                           synonyms_filename: str,
                           expansions: tuple[bool, bool],
                           checkpoint: str):
    """Translates English and evaluates against reference translations from a .JSONL file.
    Each line should have an "en" and "es" key.
    Translates, does any processing and evaluates with 3 metrics

    Writes the output CSV to a file with the following columns:
    - 'original': the original English sentence
    - 'reference': the actual provided translation
    - 'translation': the translation from the model
    - 'sacrebleu': the SacreBLEU score of the translation
    - 'ter': the translation error rate of the translation
    - 'semsim': the semantic similarity score of the translation

    :param input_file: the input file to translate, a .txt file with each sentence on a new line
    :param output_file: the output file to write results to
    :param abbreviations_filename: the filename of the abbreviations dictionary
    :param synonyms_filename: the filename of the synonyms dictionary
    :param expansions: flags for pre- and post- abbreviation expansion
    :param checkpoint: the name of the model checkpoint to use"""
    english_texts, spanish_references = [], []
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            english_texts.append(entry['en'])
            spanish_references.append(entry['es'])

    spanish_outputs = translate_english_inputs(english_texts,
                                               abbreviations_filename,
                                               synonyms_filename,
                                               expansions,
                                               checkpoint)
    data = {'english': english_texts, 'reference': spanish_references, 'translation': spanish_outputs}
    df = pd.DataFrame(data)
    df = evaluate_translations(df)
    df.to_csv(output_file, index=False, header=True)


def translate_english_inputs(english_inputs: list[str],
                             abbreviations_filename: str,
                             synonyms_filename: str,
                             expansions: tuple[bool, bool],
                             checkpoint: str):
    """Translates English inputs to Spanish outputs. Applies pre-processing, the model and post-processing."""
    # Preprocessing
    abbreviations = Abbreviations(abbreviations_filename, expansions[0], expansions[1])
    english_inputs = abbreviations.preprocess(english_inputs)

    # Translation with model
    spanish_outputs = translate_text(english_inputs, checkpoint)

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

    def ter(row):
        score = SimilarityMetric.TER.evaluate(row['reference'], row['translation'])
        return round(score * 100, round_digits)

    def semsim(row):
        score = SimilarityMetric.SEMANTIC_SIMILARITY.evaluate(row['reference'], row['translation'])
        return round(score * 100, round_digits)

    print("Evaluating translations...")
    df['sacrebleu'] = df.apply(sacrebleu, axis=1)
    df['`ter`'] = df.apply(ter, axis=1)
    df['semsim'] = df.apply(semsim, axis=1)
    return df


def main(input_filename: str,
         output_filename: str,
         evaluate: bool,
         expansion: tuple[bool, bool]):
    """Main function to translate (and evaluate) English to Spanish."""
    abbreviations_filename = "../../processing/dictionaries/processed/abbreviations.jsonl"
    synonyms_filename = "../../processing/dictionaries/processed/preferred_synonyms_es.jsonl"
    checkpoint = "za17/helsinki-biomedical-finetuned"
    if evaluate:
        translate_and_evaluate(input_filename, output_filename, abbreviations_filename,
                               synonyms_filename, expansion, checkpoint)
    else:
        translate_no_evaluate(input_filename, output_filename, abbreviations_filename,
                              synonyms_filename, expansion, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English to Spanish and evaluate translations")
    parser.add_argument("input_file", type=str, help="The input file to translate, .txt or .jsonl format")
    parser.add_argument("output_file", type=str, help="The output file to write results to")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate translations against reference translations")
    parser.add_argument("--preexpansion", action="store_true", help="Pre-expand abbreviations in Spanish")
    parser.add_argument("--postexpansion", action="store_true", help="Post-expand abbreviations in Spanish")
    args = parser.parse_args()
    print("CLI arguments:", args)
    expansion_flags = (args.preexpansion, args.postexpansion)
    main(args.input_file, args.output_file, args.evaluate, expansion_flags)
