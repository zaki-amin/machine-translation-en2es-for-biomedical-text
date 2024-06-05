import argparse

from translation.hpo.official.official_evaluator import translate_and_evaluate


def main():
    """Command line interface for evaluating model translations against official translations
    Usage: python main.py <HPO ID> [--labels]"""
    parser = argparse.ArgumentParser(
        description="Enter a HPO ID and evaluate model translation on all descendants"
    )
    parser.add_argument("hpo_id", type=str, help="HPO ID in the form HP:XXXXXXX")
    parser.add_argument("checkpoint", type=str, help="Model checkpoint")
    args = parser.parse_args()
    translate_and_evaluate(args.hpo_id, args.checkpoint)


if __name__ == "__main__":
    main()
