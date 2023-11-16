from evaluation.official.expected_translations import read_official_translations
from evaluation.official.string_similarity import SimilarityMetric
from evaluation.translation.translation_model import model_results
import pandas as pd


def evaluate_translation(hpo_id: str, spreadsheet: str, labels: bool):
    print("---Generating model translations---")
    model_df = model_results(hpo_id, "../official", labels, spreadsheet)

    print("---Reading official translations---")
    official_df = read_official_translations()

    merged_df = combine_translations(model_df, official_df, SimilarityMetric.BLEU)
    display_accuracy(merged_df)
    save_to_csv(merged_df)


def combine_translations(model_df: pd.DataFrame, official_df: pd.DataFrame, metric: SimilarityMetric):
    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    merged_df['match'] = merged_df.apply(lambda row: metric.evaluate(row['etiqueta oficial'], row['traducción modelo']),
                                         axis=1)
    return merged_df


def display_accuracy(df):
    num_translations = df.shape[0]
    hits = df['match'].sum()
    model_accuracy = hits / num_translations
    print(f"Hits: {hits}")
    print(f"Number of translations: {num_translations}")
    print("Model accuracy: {:.2%}".format(model_accuracy))


def save_to_csv(df):
    file_path = 'model_evaluation.csv'
    df.to_csv(file_path, index=False)
    print(f'CSV file "{file_path}" has been created.')
