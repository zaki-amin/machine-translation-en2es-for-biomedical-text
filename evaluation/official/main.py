import argparse

from evaluation.official.official_evaluator import evaluate_translation


def main():
    parser = argparse.ArgumentParser(
        description="Enter a HPO ID and evaluate model translation on all descendants"
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    parser.add_argument('--labels', action='store_true', default=False)
    parser.add_argument('--xlsx', type=str, help="XLSX file to read model translations from")
    args = parser.parse_args()
    evaluate_translation(args.hpo_id, args.xlsx, args.labels)


if __name__ == "__main__":
    main()
