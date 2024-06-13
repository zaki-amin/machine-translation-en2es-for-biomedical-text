import argparse
import json

import pandas as pd

from evaluations.similarity.sentence_similarity import SentenceSimilarity
from processing.abbreviations import Abbreviations
from processing.preferred_synonyms import PreferredSynonyms
from translation.translate import translate_text


def translate_no_evaluate(input_file: str,
                          output_file: str,
                          abbreviations: Abbreviations,
                          synonyms: PreferredSynonyms | None,
                          checkpoint: str):
    """Translates English from a .txt file to Spanish.
    Each line should have an English sentence.
    Applies pre-processing, the model and post-processing.
    Writes the output CSV to a file with the following columns:
    - 'en': the original English sentence
    - 'es': the model's translation"""
    if ".txt" in input_file:
        english_inputs = read_text_file(input_file)
    else:
        is_jsonl = True
        english_inputs, spanish_references = read_jsonl_file(input_file)

    spanish_outputs = translate_english_inputs(english_inputs,
                                               abbreviations,
                                               synonyms,
                                               checkpoint)
    data = {'english': english_inputs, 'translation': spanish_outputs}
    if is_jsonl:
        data['reference'] = spanish_references
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, header=True)


def read_text_file(input_file: str):
    with open(input_file, 'r') as file:
        english_inputs = [line.strip() for line in file.readlines()]
    return english_inputs


def read_jsonl_file(input_file: str):
    english_texts, spanish_references = [], []
    with open(input_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            english_texts.append(entry['en'])
            spanish_references.append(entry['es'])
    return english_texts, spanish_references


def translate_and_evaluate(input_file: str,
                           output_file: str,
                           abbreviations: Abbreviations,
                           synonyms: PreferredSynonyms | None,
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
    - 'semsim': the semantic similarity score of the translation"""
    english_texts, spanish_references = read_jsonl_file(input_file)

    spanish_outputs = translate_english_inputs(english_texts,
                                               abbreviations,
                                               synonyms,
                                               checkpoint)
    data = {'english': english_texts, 'reference': spanish_references, 'translation': spanish_outputs}
    df = pd.DataFrame(data)
    df = evaluate_translations(df)
    df.to_csv(output_file, index=False, header=True)


def translate_english_inputs(english_inputs: list[str],
                             abbreviations: Abbreviations,
                             synonyms: PreferredSynonyms | None,
                             checkpoint: str):
    """Translates English inputs to Spanish outputs. Applies pre-processing, the model and post-processing."""
    # Preprocessing
    english_inputs = abbreviations.preprocess(english_inputs)

    # Translation with model
    spanish_outputs = translate_text(english_inputs, checkpoint)

    # Postprocessing
    if synonyms is not None:
        print("Postprocessing with synonyms...")
        spanish_outputs = synonyms.postprocess(spanish_outputs)
    spanish_outputs = abbreviations.postprocess(spanish_outputs)
    return spanish_outputs


def evaluate_translations(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates the translations, adding metrics to the dataframe.
    The new columns are 'sacrebleu' for the SacreBLEU score, 'semsim' for the semantic similarity score
    and 'wer' for the word error rate."""

    round_digits = 1

    def sacrebleu(row):
        score = SentenceSimilarity.SACREBLEU.evaluate(row['reference'], row['translation'])
        return round(score, round_digits)

    def ter(row):
        score = SentenceSimilarity.TER.evaluate(row['reference'], row['translation'])
        return round(score, round_digits)

    def semsim(row):
        score = SentenceSimilarity.SEMANTIC_SIMILARITY.evaluate(row['reference'], row['translation'])
        return round(score * 100, round_digits)

    print("Evaluating translations...")
    df['sacrebleu'] = df.apply(sacrebleu, axis=1)
    df["'ter'"] = df.apply(ter, axis=1)
    df['semsim'] = df.apply(semsim, axis=1)
    return df


def main(input_filename: str,
         output_filename: str,
         evaluate: bool,
         abbreviations: Abbreviations,
         synonyms: PreferredSynonyms | None):
    """Main function to translate (and evaluate) English to Spanish."""
    checkpoint = "za17/helsinki-biomedical-finetuned"
    if evaluate:
        translate_and_evaluate(input_filename, output_filename, abbreviations, synonyms, checkpoint)
    else:
        translate_no_evaluate(input_filename, output_filename, abbreviations, synonyms, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English to Spanish and evaluate translations")
    parser.add_argument("input_file", type=str, help="The input file to translate, .txt or .jsonl format")
    parser.add_argument("output_file", type=str, help="The output file to write results to")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate translations against reference translations")
    parser.add_argument("--preexpansion", action="store_true", help="Pre-expand abbreviations in English")
    parser.add_argument("--postexpansion", action="store_true", help="Post-expand abbreviations in Spanish")
    parser.add_argument("--synonyms", action="store_true", help="Replace secondary synonyms with primary synonyms")

    args = parser.parse_args()
    print("CLI arguments:", args)

    abbreviations = Abbreviations("../../processing/dictionaries/processed/abbreviations.jsonl",
                                  args.preexpansion, args.postexpansion)
    synonyms = None if not args.synonyms else PreferredSynonyms("../../processing/dictionaries/processed"
                                                                "/preferred-synonyms-es.jsonl")
    main(args.input_file, args.output_file, args.evaluate, abbreviations, synonyms)
