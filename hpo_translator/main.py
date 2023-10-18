import argparse

from src.translate import translate_hpo, translate_labels


def normal_function():
    parser = argparse.ArgumentParser(
        description="Translates an HPO term and its descendants."
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    args = parser.parse_args()

    translate_hpo(args.hpo_id)


def only_labels():
    parser = argparse.ArgumentParser(
        description="Translates an HPO term and its descendants."
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    args = parser.parse_args()

    translate_labels(args.hpo_id)


if __name__ == "__main__":
    only_labels()
