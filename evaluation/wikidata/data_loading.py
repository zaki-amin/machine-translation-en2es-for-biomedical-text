import json
from dataclasses import dataclass
from typing import Set, Dict


@dataclass
class WikidataHPO:
    id: str
    label: str
    etiqueta: str
    synonyms: Set[str]
    sinónimas: Set[str]


def parse_wikidata_json(filename: str) -> Dict[str, WikidataHPO]:
    with open(filename, "r") as file:
        data = json.load(file)

    hpo_objects = {}
    for hpo_id in data:
        hpo_objects[hpo_id] = WikidataHPO(id=hpo_id,
                                          label=data[hpo_id]["term_en"],
                                          etiqueta=data[hpo_id]["term_es"],
                                          synonyms=set(data[hpo_id]["synonyms_en"]),
                                          sinónimas=set(data[hpo_id]["synonyms_es"]))
    return hpo_objects


if __name__ == "__main__":
    print(parse_wikidata_json("sample.json"))
