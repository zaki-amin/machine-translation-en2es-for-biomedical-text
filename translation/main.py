import argparse

from src.translate import translate_hpo


def main():
    parser = argparse.ArgumentParser(
        description="Translates an HPO term and its descendants."
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    parser.add_argument('--labels', action='store_true', default=False)
    args = parser.parse_args()

    translate_hpo(args.hpo_id, only_labels=args.labels)


if __name__ == "__main__":
    main()
