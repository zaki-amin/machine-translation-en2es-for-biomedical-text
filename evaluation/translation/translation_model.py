import pandas as pd
import os

from evaluation.official.expected_translations import clean_column
from translation.src.translate import translate_hpo


def model_results(hpo_id: str, labels: bool) -> pd.DataFrame:
    filename = f"/Users/zaki/PycharmProjects/hpo_evaluation/files/model_translations/{hpo_id}.xlsx"
    if os.path.isfile(filename):
        print("---Reading model translations---")
    else:
        print("---Generating model translations---")
        translate_hpo(hpo_id, only_labels=labels)

    translation_df = pd.read_excel(filename, sheet_name='Sheet1')
    translation_df = translation_df.rename(columns={'id': 'hpo_id', 'spanish': 'traducción modelo'})
    return clean_column(translation_df, 'traducción modelo')
