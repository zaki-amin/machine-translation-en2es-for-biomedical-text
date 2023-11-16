import pandas as pd

from hpo_translator.src.translate import translate_hpo


def model_results(hpo_id: str, out_dir: str | None, labels: bool, spreadsheet: str | None) -> pd.DataFrame:
    if spreadsheet is None:
        print("---Generating model translations---")
        translate_hpo(hpo_id, out_dir=out_dir, only_labels=labels)
        result_file = 'hpo_translation.xlsx'
    else:
        print("---Reading model translations---")
        result_file = spreadsheet

    model_df = pd.read_excel(result_file, sheet_name='Sheet1')
    model_df = model_df.rename(columns={'id': 'hpo_id', 'spanish': 'traducción modelo'})
    return model_df
