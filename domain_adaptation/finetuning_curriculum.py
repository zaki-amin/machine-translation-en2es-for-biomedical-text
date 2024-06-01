import torch

from domain_adaptation.corpus import load_corpus
from domain_adaptation.finetuning import login_and_get_repo
from domain_adaptation.finetuning_trainer import FineTuningTrainer


def main(hf_token: str,
         train_filepath: str,
         epochs_per_corpus: int,
         lrs: dict[str, float],
         train_batch_size: int,
         eval_batch_size: int,
         device: str):
    model_name, repo = login_and_get_repo(hf_token)
    trainer_fine_tuning = FineTuningTrainer("Helsinki-NLP/opus-mt-en-es", device)

    # ordered by target vocabulary size from smallest to largest
    ordered_corpora = ["khresmoi-tr", "orphanet-terms", "clinspen-tr", "medline", "preferred-en2es", "snomed",
                       "orphanet-definitions-tr", "pubmed-tr"]

    # ordered by target length from smallest to largest
    # ordered_corpora = ["preferred-en2es", "orphanet-terms", "clinspen-tr", "snomed", "medline", "khresmoi-tr",
    # "pubmed-tr", "orphanet-definitions-tr"]

    for filename in ordered_corpora:
        full_filename = train_filepath + filename + ".jsonl"
        corpus = load_corpus(full_filename, 0.1, 42)
        print(f"Training on {filename}")
        trainer_fine_tuning.finetune_with_trainer(corpus,
                                                  model_name,
                                                  lrs[filename],
                                                  train_batch_size,
                                                  eval_batch_size,
                                                  epochs_per_corpus)


if __name__ == "__main__":
    train_directory = "/home/zakiamin/PycharmProjects/hpo_translation/corpus/train/"
    token = "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"
    seed = 17
    torch.manual_seed(seed)
    epochs_per_corpus, batch_size = 3, 8
    # smaller datasets with larger learning rates

    lrs = {"khresmoi-tr": 1e-5,
           "orphanet-definitions-tr": 5e-6,
           "pubmed-tr": 1e-6,
           "orphanet-terms": 5e-7,
           "medline": 3e-7,
           "clinspen-tr": 3e-7,
           "snomed": 1e-7,
           "preferred-en2es": 5e-8}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    main(token, train_directory, epochs_per_corpus, lrs, batch_size, batch_size * 2, device)
