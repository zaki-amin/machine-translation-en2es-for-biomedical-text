import pandas as pd

from evaluation.similarity.sentence_similarity import SentenceSimilarity
from translation.hpo.official.translations import read_official_translations
from translation.translate import translate_hpo


def translate_and_evaluate(hpo_id: str, checkpoint: str):
    """Generates model translations and evaluates against official translations for a given HPO ID.
    Saves results to <hpo.id>.csv
    :param hpo_id: HPO ID in the form HP:XXXXXXX
    :param checkpoint: Model checkpoint"""
    print("---Generating translations---")
    translation_df = translate_hpo(hpo_id, checkpoint)

    official_df = read_official_translations("official/hp-es.babelon.tsv", '\t')
    print("---Comparing translations---")
    merged_df = compare_translations(translation_df, official_df)
    merged_df.to_csv(f"{hpo_id}.csv", index=False)


def compare_translations(model_df: pd.DataFrame, official_df: pd.DataFrame) -> pd.DataFrame:
    """Combines model translations with official translations and evaluates all metrics
    :return: Combined dataframe with a new column for each similarity metric"""
    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    for metric in SentenceSimilarity:
        merged_df[str(metric)] = merged_df.apply(
            lambda row: metric.evaluate(row['reference'], row['translation']),
            axis=1)

    return merged_df
