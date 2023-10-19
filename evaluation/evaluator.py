from enum import Enum

from evaluation.expected_translations import read_official_translations
from hpo_translator.src.translate import translate_hpo
import pandas as pd


def evaluate_translation(hpo_id: str, spreadsheet: str, labels: bool):
    if spreadsheet is None:
        print("---Generating model translations---")
        translate_hpo(hpo_id, only_labels=labels)
        result_file = 'hpo_translation.xlsx'
    else:
        print("---Reading model translations---")
        result_file = spreadsheet

    model_df = pd.read_excel(result_file, sheet_name='Sheet1')
    model_df = model_df.rename(columns={'id': 'hpo_id', 'spanish': 'traducción modelo'})

    print("---Reading official translations---")
    official_df = read_official_translations()

    merged_df = combine_translations(model_df, official_df)
    print_accuracy(merged_df)
    save_to_csv(merged_df)


def combine_translations(model_df, official_df):
    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    merged_df['match'] = merged_df['traducción modelo'] == merged_df['etiqueta oficial']
    print(merged_df)
    return merged_df


def print_accuracy(df):
    num_translations = df.shape[0]
    model_accuracy = df['match'].value_counts()[True] / num_translations
    print("Model accuracy: {:.2%}".format(model_accuracy))


def save_to_csv(df):
    file_path = 'model_evaluation.csv'
    df.to_csv(file_path, index=False)
    print(f'CSV file "{file_path}" has been created.')


class TranslateChoice(Enum):
    LABELS = 1
    DEFINITIONS = 2
    SYNONYMS = 3
    ALL = 4

    def attribute_list(self):
        match self:
            case TranslateChoice.LABELS:
                return ["name"]
            case TranslateChoice.DEFINITIONS:
                return ["definition"]
            case TranslateChoice.SYNONYMS:
                return ["synonyms"]
            case TranslateChoice.ALL:
                return ["name", "definition", "synonyms", "comment"]



