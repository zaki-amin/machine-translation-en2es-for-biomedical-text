import pandas as pd

from evaluation.translation.translation_model import model_results
from evaluation.wikidata.data_loading import parse_wikidata_json, WikidataHPO


def evaluate_translation(hpo_id: str, just_labels: bool, spreadsheet: str):
    model_df: pd.DataFrame = model_results(hpo_id, just_labels, spreadsheet)
    wikidata_translations: dict[str, WikidataHPO] = parse_wikidata_json("sample.json")

    name_hits, total_names = 0, 0
    synonym_hits, total_synonyms = 0, 0

    for index, row in model_df.iterrows():
        id = row['hpo_id']
        model_translation = row['traducción modelo']
        wikidata_translation = wikidata_translations.get(id, None)

        match row['kind']:
            case 'name':
                total_names += 1
                if wikidata_translation is not None and model_translation == wikidata_translation.etiqueta:
                    name_hits += 1
            case 'synonym':
                total_synonyms += 1
                if wikidata_translation is not None and model_translation in wikidata_translation.sinónimas:
                    synonym_hits += 1

    print(name_hits, total_names)
    print(synonym_hits, total_synonyms)


if __name__ == "__main__":
    evaluate_translation("HP:0000024", True, "../hpo_translation.xlsx")
