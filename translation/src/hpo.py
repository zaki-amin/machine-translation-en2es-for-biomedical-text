from pathlib import Path

import polars as pl
import pronto
from torch.utils.data import Dataset


def get_subclasses(hpo: pronto.Ontology, hpo_id: str, with_self: bool = True):
    """
    Obtain a list of IDs of children terms.
    """
    term = hpo.get_term(hpo_id)
    return [getattr(node, "id") for node in term.subclasses(with_self=with_self)]


def term_text(hpo: pronto.Ontology, hpo_id: str, just_labels: bool):
    """
    Obtains a dictionary with the attributes containing text.
    """
    term_dict = {}
    term = hpo.get_term(hpo_id)
    attributes = ["name"] if just_labels else ("name", "definition", "comment", "synonyms")
    for k in attributes:
        if k != "synonyms":
            term_dict[k] = str(getattr(term, k))
        else:
            term_dict[k] = [syn.description for syn in term.synonyms]

    return term_dict


def collate_fn(batch):
    return [list(elem) for elem in zip(*batch)]


class HPOCorpus(Dataset):
    """
    PyTorch dataset for the Human Phenotype Ontology.

    Args:
        hpo_id: str
            ID of the highest node, all its children will
            be included.
    """

    def __init__(self, hpo_id: str = "HP:0000001", just_labels: bool = False):
        # Load the ontology
        hpo = pronto.Ontology.from_obo_library("hp.obo")

        # Get the IDs and terms to include
        self._prep_terms(hpo, hpo_id, just_labels)
        self.trans = pl.DataFrame(schema={"index": pl.Int64, "text": pl.Utf8, "kind": pl.Utf8})

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, idx):
        return self.terms[idx, "index"], self.terms[idx, "text"]

    def _prep_terms(self, hpo: pronto.Ontology, hpo_id: str, just_labels: bool):
        """
        Prepare terms as dictionaries of strings, with list items
        expanding as multiple keys.
        """
        # HPO IDs
        self.ids = get_subclasses(hpo, hpo_id)

        terms = []
        for hpo_id in self.ids:
            term_dict = term_text(hpo, hpo_id, just_labels)
            term = {}
            for key, val in term_dict.items():
                if isinstance(val, list):
                    term.update({f"{key}_{i}": elem for i, elem in enumerate(val)})
                if isinstance(val, str):
                    term[key] = val
            terms.append(
                {"id": hpo_id, "header": list(term.keys()), "text": list(term.values())}
            )
        self.terms = (
            pl.DataFrame(terms)
            .explode(["header", "text"])
            .with_row_count("index")
            .with_columns(pl.col("index").cast(pl.Int64))
        )

    def set_trans(self, idxs: int | list[int], text: str | list[str], kind: str | list[str] = "definition"):
        self.trans.extend(pl.DataFrame({"index": idxs, "text": text, "kind": kind}))

    def save_pairs(self, out_dir: str, filename: str = "hpo_translation.xlsx"):
        pairs = self.translation_pairs()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        excel_path = Path(out_dir, filename).as_posix()
        pairs.write_excel(excel_path, autofit=True)

    def translation_pairs(self):
        return (
            self.terms.join(self.trans, on=["index"], suffix="_trans")
            .rename({"text": "english", "text_trans": "spanish"})
            .select(["id", "kind", "english", "spanish"])
            .filter(pl.col("english").str.strip().str.to_lowercase() != "none")
        )
