import pandas as pd

from hpo.discrepancy.word_similarity import word_differences, most_similar_word
from hpo.official.string_similarity import SimilarityMetric
from hpo.utility.text_functions import trim_string


def drop_similarity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Drops all similarity metrics from the given dataframe"""
    for metric in SimilarityMetric:
        df = df.drop(metric.name, axis=1)
    return df


def scan_all_translations(hpo_id: str):
    """Opens translation file for the given HPO ID, prints the most similar words for those which have been
    mistranslated and saves the differences between official and model to a CSV file.
    """
    translation_filepath = f"/Users/zaki/PycharmProjects/hpo_evaluation/files/results/official/{hpo_id}.csv"
    df = pd.read_csv(translation_filepath)

    similarities = find_most_similar_words(df)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for word, (most_similar, similarity_score) in sorted_similarities:
        print(f"{word} -> {most_similar} (similarity score: {similarity_score})")

    df = append_word_differences(df)
    df = drop_rows_with_no_differences(df)
    df = drop_similarity_metrics(df)

    difference_filepath = f"/Users/zaki/PycharmProjects/hpo_evaluation/files/results/differences/{hpo_id}.csv"
    df.to_csv(difference_filepath, index=False)


def find_most_similar_words(df: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Iterates through the translation dataframe and returns a similarity dictionary.
    This similarity dictionary maps each mistranslated model word to a tuple of the most similar word
    in the official translation and the corresponding similarity score from spacy.


    :param df: The translation dataframe which must have the columns 'traducción modelo' and 'etiqueta oficial'
    :return: A dictionary from each word to its most similar word and the similarity score from spacy.
    Only contains all words translated by a model which do not appear in the official translation.
    """
    word_similarities: dict[str, tuple[str, float]] = {}
    df.apply(lambda row: compare_row_and_update_similarities(row, word_similarities), axis=1)
    return word_similarities


def compare_row_and_update_similarities(row: pd.Series, word_similarities: dict[str, tuple[str, float]]):
    """Compares the official term and the model term in the given row and updates the word similarities dictionary
    :param row: A row from the translation dataframe
    :param word_similarities: The word similarities dictionary
    """
    official_term = trim_string(row["etiqueta oficial"])
    model_term = trim_string(row["traducción modelo"])

    if official_term == model_term:
        return

    missing_official, missing_model = word_differences(official_term, model_term)
    for mistranslation in missing_model:
        similar_word, similar_score = most_similar_word(mistranslation, official_term)
        if mistranslation not in word_similarities or similar_score > word_similarities[mistranslation][1]:
            word_similarities[mistranslation] = (similar_word, similar_score)


def append_word_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Appends two columns to the given dataframe: 'missing official' and 'missing model'.
    The 'missing official' column contains all words in the official translation which are not in the model translation.
    The 'missing model' column contains all words in the model translation which are not in the official translation.

    :param df: The translation dataframe which must have the columns 'traducción modelo' and 'etiqueta oficial'
    :return: The given dataframe with the two additional columns
    """
    df["missing official"] = df.apply(
        lambda row: word_differences(row["etiqueta oficial"], row["traducción modelo"])[0],
        axis=1)
    df["missing model"] = df.apply(lambda row: word_differences(row["etiqueta oficial"], row["traducción modelo"])[1],
                                   axis=1)
    return df


def drop_rows_with_no_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Drops all rows from the given dataframe where the model and official translations are equivalent"""
    condition = (df["missing official"].apply(len) == 0) & (df["missing model"].apply(len) == 0)
    return df.drop(df[condition].index)


if __name__ == "__main__":
    scan_all_translations("HP:0001197")
