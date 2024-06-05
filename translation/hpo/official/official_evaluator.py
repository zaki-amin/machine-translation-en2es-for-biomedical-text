import pandas as pd

from evaluations.sentence_similarity import SimilarityMetric
from translation.hpo.official.expected_translations import read_official_translations, clean_column
from translation.translate import translate_hpo


def translate_and_evaluate(hpo_id: str, checkpoint: str):
    """Generates model translations and evaluates against official translations for a given HPO ID.
    Saves results to CSV in directory results/.
    :param hpo_id: HPO ID in the form HP:XXXXXXX
    :param checkpoint: Model checkpoint"""
    model_df = generate_model_translations(hpo_id, checkpoint)

    official_df = read_official_translations("official/hp-es.babelon.tsv", '\t')

    print("---Comparing translations---")
    merged_df = compare_translations(model_df, official_df)
    display_accuracy(merged_df)
    # Drop the 'kind' column
    merged_df = merged_df.drop(columns=['kind'])
    merged_df.to_csv(f"results/{hpo_id}.csv", index=False)


def generate_model_translations(hpo_id: str, checkpoint: str) -> pd.DataFrame:
    print("---Generating model translations---")
    translate_hpo(hpo_id, checkpoint)
    translation_df = pd.read_excel(f"results/{hpo_id}.xlsx", sheet_name='Translations')
    translation_df = translation_df.rename(columns={'id': 'hpo_id', 'spanish': 'traducción modelo'})
    return clean_column(translation_df, 'traducción modelo')


def compare_translations(model_df: pd.DataFrame, official_df: pd.DataFrame) -> pd.DataFrame:
    """Combines model translations with official translations and evaluates all metrics
    :return: Combined dataframe with a new column for each similarity metric"""
    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    for metric in SimilarityMetric:
        merged_df[metric.name] = merged_df.apply(
            lambda row: metric.evaluate(row['etiqueta oficial'], row['traducción modelo']),
            axis=1)

    merged_df = merged_df.rename(columns={'SEMANTIC_SIMILARITY': 'SEMSIM'})
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
            model_performance = model_performance / 100
        if metric == SimilarityMetric.TER:
            model_performance = model_performance / 10 * -1
        print("Model performance: {:.2%}".format(model_performance))
