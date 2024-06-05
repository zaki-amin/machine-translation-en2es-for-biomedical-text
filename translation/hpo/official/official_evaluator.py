import os

from translation.hpo.official.expected_translations import read_official_translations, clean_column
from evaluations.sentence_similarity import SimilarityMetric
import pandas as pd

from translation.translate import translate_hpo


def evaluate_translation(hpo_id: str, checkpoint: str, labels: bool):
    """Compares model translations against official translations for a given HPO ID. Saves results to CSV in directory files/results.
    :param hpo_id: HPO ID in the form HP:XXXXXXX
    :param checkpoint: Model checkpoint
    :param labels: True if evaluating only labels, False if evaluating everything e.g. synonyms"""
    model_df = gather_model_translations(hpo_id, checkpoint, labels)

    official_df = read_official_translations("official/hp-es.babelon.tsv", '\t')

    print("---Comparing translations---")
    merged_df = combine_translations(model_df, official_df)
    display_accuracy(merged_df)
    merged_df.to_csv(f"results/{hpo_id}.csv", index=False)


def gather_model_translations(hpo_id: str, checkpoint: str, labels: bool) -> pd.DataFrame:
    print("---Generating model translations---")
    translate_hpo(hpo_id, checkpoint, only_labels=labels)
    translation_df = pd.read_excel(f"results/{hpo_id}.xlsx", sheet_name='Translations')
    translation_df = translation_df.rename(columns={'id': 'hpo_id', 'spanish': 'traducción modelo'})
    return clean_column(translation_df, 'traducción modelo')


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
        if metric == SimilarityMetric.SACREBLEU:
            # Uses a 0-100 scale instead of 0-1
            score = score / 100
            model_performance = model_performance / 100
        print(f"Score: {score} / {num_translations}")
        print("Model performance: {:.2%}".format(model_performance))
