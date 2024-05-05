import pandas as pd

from dictionaries.abbreviations import Abbreviations
from evaluation.official.string_similarity import SimilarityMetric
from hpo_translator.src.models import MarianMTConfig, NMTModelConfig
from hpo_translator.src.translate import translate


def translate_input(input_file: str,
                    output_file: str,
                    checkpoint: str | None = None,
                    config: NMTModelConfig = MarianMTConfig()):
    """Translates the input .txt file from English to Spanish.
    Applies pre-processing, the model and post-processing.
    Writes the output CSV to a file with the following columns:
    - 'original': the original English sentence
    - 'translation': the actual provided translation

    :param input_file: the input file to translate, a .txt file with each sentence on a new line
    :param output_file: the output file to write results to
    :param checkpoint: the name of the model checkpoint to use. If not given, defaults to clinical MarianMT
    :param config: the configuration of the model to use. If not given, defaults to MarianMTConfig"""
    with open(input_file, 'r') as file:
        english_inputs = [line.strip() for line in file.readlines()]

    # Preprocessing
    abbreviations = Abbreviations(
        "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/abbreviations.jsonl")
    english_inputs = abbreviations.preprocess(english_inputs)

    # Translation with model
    translated = translate(english_inputs, checkpoint, config)

    # Postprocessing
    synonyms_filename = "/Users/zaki/PycharmProjects/hpo_translation/dictionaries/processed/synonyms.jsonl"
    # translated = postprocess(translated, synonyms_filename)

    write_output(english_inputs, translated, output_file)


def preprocess(english_inputs: list[str], abbreviations_filename: str) -> list[str]:
    abbreviations = Abbreviations(abbreviations_filename)
    return list(map(lambda line: abbreviations.expand_all_abbreviations_english(line), english_inputs))


def postprocess(df: pd.DataFrame, synonyms_filename: str) -> list[str]:
    pass


def write_output(sources: list[str], targets: list[str], output_file: str):
    """Writes the English sources and Spanish targets to a CSV file."""
    data = {'en': sources, 'es': targets}
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, header=True)


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


def main():
    translate_input("input.txt", "output.csv")


if __name__ == "__main__":
    main()
