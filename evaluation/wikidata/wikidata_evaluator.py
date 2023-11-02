from typing import Dict

import pandas as pd

from evaluation.translation.translation_model import model_results
from evaluation.wikidata.data_loading import parse_wikidata_json, WikidataHPO


def evaluate_translation(hpo_id: str, spreadsheet: str, labels: bool):
    model_df: pd.DataFrame = model_results(hpo_id, labels, spreadsheet)
    # wikidata_info: Dict[str, WikidataHPO] = parse_wikidata_json("queries/translated_dataset.json")
    #
    # for hpo_id in model_df['hpo_id']:



