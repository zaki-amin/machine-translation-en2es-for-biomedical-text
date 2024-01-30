from evaluation.official.expected_translations import read_official_translations
from evaluation.official.string_similarity import SimilarityMetric
from evaluation.translation.translation_model import model_results
import pandas as pd

from evaluation.utility.file_functions import save_to_csv


def evaluate_translation(hpo_id: str, labels: bool):
    """Compares model translations against official translations for a given HPO ID. Saves results to CSV in directory files/results.
    :param hpo_id: HPO ID in the form HP:XXXXXXX
    :param labels: True if evaluating only labels, False if evaluating everything e.g. synonyms"""
    model_df = model_results(hpo_id, labels)

    print("---Reading official translations---")
    official_df = read_official_translations(
        "/Users/zaki/Desktop/Estudios/Master's thesis/Resources/HPO data/hp-es.babelon.tsv", '\t')

    print("---Comparing translations---")
    merged_df = combine_translations(model_df, official_df)
    display_accuracy(merged_df)
    save_to_csv(merged_df, "/Users/zaki/PycharmProjects/hpo_evaluation/files/results/official", hpo_id)


def combine_translations(model_df: pd.DataFrame, official_df: pd.DataFrame) -> pd.DataFrame:
    """Combines model translations with official translations and evaluates all metrics
    :return: Combined dataframe with a new column for each similarity metric"""
    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    for metric in SimilarityMetric:
        merged_df[metric.name] = merged_df.apply(
            lambda row: metric.evaluate(row['etiqueta oficial'], row['traducción modelo']),
            axis=1)

    return merged_df


def display_accuracy(df: pd.DataFrame):
    """For each metric, print the model's performance against the official translations"""
    num_translations = df.shape[0]
    print(f"Number of translations: {num_translations}")

    for metric in SimilarityMetric:
        print(f"\nMetric: {metric.name}")
        score = df[metric.name].sum()
        model_performance = score / num_translations
        print(f"Score: {score} / {num_translations}")
        print("Model performance: {:.2%}".format(model_performance))
