import torch

from domain_adaptation.corpus import load_corpus
from domain_adaptation.finetuning import login_and_get_repo
from domain_adaptation.finetuning_trainer import FineTuningTrainer


def main(hf_token: str,
         train_filepath: str,
         epochs_per_corpus: int,
         learning_rates: dict[str, float],
         train_batch_size: int,
         eval_batch_size: int):
    model_name, repo = login_and_get_repo(hf_token)
    trainer_fine_tuning = FineTuningTrainer("Helsinki-NLP/opus-mt-en-es")

    # ordered by target vocabulary size from smallest to largest
    ordered_corpora = ["khresmoi-tr", "orphanet-terms", "clinspen-tr", "medline", "dptm", "snomed",
                       "orphanet-definitions-tr", "pubmed-tr"]

    for filename in ordered_corpora:
        full_filename = train_filepath + filename + ".jsonl"
        corpus = load_corpus(full_filename, 0.1, 42)
        print(f"Training on {filename}")
        trainer_fine_tuning.finetune_with_trainer(corpus,
                                                  model_name,
                                                  learning_rates[filename],
                                                  train_batch_size,
                                                  eval_batch_size,
                                                  epochs_per_corpus)


if __name__ == "__main__":
    train_directory = "../corpus/train/"
    token = input("Enter Hugging Face API token: ")
    seed = 17
    torch.manual_seed(seed)
    corpus_epochs, batch_size = 3, 8

    # smaller datasets with larger learning rates
    lrs = {"khresmoi-tr": 1e-5,
           "orphanet-definitions-tr": 8e-6,
           "pubmed-tr": 2e-7,
           "orphanet-terms": 5e-7,
           "medline": 7e-7,
           "clinspen-tr": 6e-7,
           "snomed": 2e-7,
           "dptm": 1e-7}

    main(token, train_directory, corpus_epochs, lrs, batch_size, batch_size * 2)
