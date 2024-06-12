import pandas as pd


def read_official_translations(file: str, delimiter: str) -> pd.DataFrame:
    """Reads official translations from file
    :param file: path to file containing official translations
    :param delimiter: delimiter used in file such as tab or comma
    :return: DataFrame with two columns: hpo_id and reference
    """
    df = pd.read_table(file, delimiter=delimiter)
    unnecessary_columns = ['source_language', 'translation_language', 'source_value',
                           'predicate_id', 'translation_status']
    df = df.drop(columns=unnecessary_columns)
    df = df.rename(columns={'subject_id': 'hpo_id', 'translation_value': 'reference'})
    return clean_column(df, 'reference')


def clean_column(translation_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cleans whitespace from the model's translations"""

    def trim_string(s: str) -> str:
        return s.strip()

    translation_df[column] = translation_df[column].apply(trim_string)
    return translation_df
