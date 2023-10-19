import argparse

from evaluation.evaluator import evaluate_translation


def main():
    parser = argparse.ArgumentParser(
        description="Enter a HPO ID and evaluate model translation on all descendants"
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    parser.add_argument('--labels', action='store_true', default=False)
    args = parser.parse_args()
    evaluate_translation(args.hpo_id, args.labels)


if __name__ == "__main__":
    main()
