import pandas as pd

from hpo.translation.translation_model import model_results
from hpo.wikidata.data_loading import parse_wikidata_json, WikidataHPO


def evaluate_translation(hpo_id: str, just_labels: bool, wiki_translations_filename: str, spreadsheet: str = None):
    """
    Evaluate model translations from the given HPO term against the Wikidata translations.
    :param hpo_id: The HPO term to begin the translations from
    :param just_labels: A flag to translate only the labels of the HPO terms
    :param wiki_translations_filename: The name of the JSON file of Wikidata translations
    :param spreadsheet: The name of the spreadsheet of model translations if already generated
    :return:
    """
    model_df: pd.DataFrame = model_results(hpo_id, just_labels)
    wikidata_translations: dict[str, WikidataHPO] = parse_wikidata_json(wiki_translations_filename)

    name_hits, name_absent, total_names = 0, 0, 0
    synonym_hits, synonym_absent, total_synonyms = 0, 0, 0
    failures = {}

    for index, row in model_df.iterrows():
        id = row['hpo_id']
        model_translation = row['traducción modelo']
        wikidata_translation = wikidata_translations.get(id, None)

        match row['kind']:
            case 'name':
                total_names += 1
                if wikidata_translation is None:
                    name_absent += 1
                else:
                    if model_translation == wikidata_translation.etiqueta:
                        name_hits += 1
                    else:
                        failures[id] = (model_translation, wikidata_translation.etiqueta)
            case 'synonym':
                if just_labels:
                    continue

                total_synonyms += 1
                if wikidata_translation is None:
                    synonym_absent += 1
                else:
                    synonym_hits += 1 if model_translation in wikidata_translation.sinónimas else 0

    print(f"Total names: {total_names}, Matches: {name_hits}, Absent: {name_absent}")
    print(f"Total synonyms: {total_synonyms}, Matches: {synonym_hits}, Absent: {synonym_absent}")
    print(f"Failures: {failures}")


if __name__ == "__main__":
    evaluate_translation("HP:0000034", True, "queries/wikidata.json", None)
