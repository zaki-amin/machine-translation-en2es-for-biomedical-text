import argparse

from evaluation.official.official_evaluator import evaluate_translation


def main():
    """Command line interface for evaluating model translations against official translations
    Usage: python main.py <HPO ID> [--labels]"""
    parser = argparse.ArgumentParser(
        description="Enter a HPO ID and evaluate model translation on all descendants"
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    parser.add_argument('--labels', action='store_true', default=False)
    args = parser.parse_args()
    evaluate_translation(args.hpo_id, args.labels)


if __name__ == "__main__":
    main()
