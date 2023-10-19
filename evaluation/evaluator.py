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
    model_df = model_df.rename(columns={'id': 'hpo_id'})

    print("---Reading official translations---")
    official_df = read_official_translations()

    merged_df = pd.merge(model_df, official_df, on='hpo_id', how='inner')
    merged_df['match'] = merged_df['spanish'] == merged_df['etiqueta']
    print(merged_df)