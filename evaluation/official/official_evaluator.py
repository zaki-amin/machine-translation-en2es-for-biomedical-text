from evaluation.official.expected_translations import read_official_translations
from evaluation.official.string_similarity import SimilarityMetric
from evaluation.translation.translation_model import model_results
import pandas as pd


def evaluate_translation(hpo_id: str, labels: bool):
    model_df = model_results(hpo_id, labels)

    print("---Reading official translations---")
    official_df = read_official_translations()

    merged_df = combine_translations(model_df, official_df)
    display_accuracy(merged_df)
    save_to_csv(merged_df, "/Users/zaki/PycharmProjects/hpo_evaluation/evaluation/results/official", hpo_id)


def combine_translations(model_df: pd.DataFrame, official_df: pd.DataFrame):
    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    for metric in SimilarityMetric:
        merged_df[metric.name] = merged_df.apply(
            lambda row: metric.evaluate(row['etiqueta oficial'], row['traducción modelo']),
            axis=1)


    return merged_df


def display_accuracy(df):
    num_translations = df.shape[0]
    print(f"Number of translations: {num_translations}")

    for metric in SimilarityMetric:
        print(f"\nMetric: {metric.name}")
        score = df[metric.name].sum()
        model_accuracy = score / num_translations
        print(f"Score: {score}")
        print("Model accuracy: {:.2%}".format(model_accuracy))


def save_to_csv(df: pd.DataFrame, folder_path: str, hpo_id: str):
    file_path = f"{folder_path}/{hpo_id}.csv"
    df.to_csv(file_path, index=False)
    print(f"\nCSV file {file_path} has been created")
