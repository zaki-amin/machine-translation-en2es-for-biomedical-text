import pandas as pd
import os

from hpo_translator.src.translate import translate_hpo


def model_results(hpo_id: str, labels: bool) -> pd.DataFrame:
    filename = f"/Users/zaki/PycharmProjects/hpo_evaluation/evaluation/model_translations/{hpo_id}.xlsx"
    if os.path.isfile(filename):
        print("---Reading model translations---")
    else:
        print("---Generating model translations---")
        translate_hpo(hpo_id, only_labels=labels)

    model_df = pd.read_excel(filename, sheet_name='Sheet1')
    model_df = model_df.rename(columns={'id': 'hpo_id', 'spanish': 'traducción modelo'})
    return model_df
