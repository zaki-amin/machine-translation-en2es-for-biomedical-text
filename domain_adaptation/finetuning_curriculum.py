import torch

from domain_adaptation.corpus import load_corpus
from domain_adaptation.finetuning import login_and_get_repo
from domain_adaptation.finetuning_trainer import FineTuningTrainer


def main(hf_token: str,
         train_filepath: str,
         epochs_per_corpus: int,
         lr: float,
         train_batch_size: int,
         eval_batch_size: int):
    model_name, repo = login_and_get_repo(hf_token)
    trainer_fine_tuning = FineTuningTrainer("Helsinki-NLP/opus-mt-en-es")

    # ordered by target vocabulary size from smallest to largest
    ordered_corpora = ["khresmoi-tr", "orphanet-terms", "clinspen-tr", "medline", "preferred-en2es", "snomed",
                       "orphanet-definitions-tr", "abstracts-tr"]

    # ordered by target length from smallest to largest
    # ordered_corpora = ["preferred-en2es", "orphanet-terms", "clinspen-tr", "snomed", "medline", "khresmoi-tr",
    # "abstracts-tr", "orphanet-definitions-tr"]

    for filename in ordered_corpora:
        full_filename = train_filepath + filename + ".jsonl"
        corpus = load_corpus(full_filename, 0.1, 42)
        trainer_fine_tuning.finetune_with_trainer(corpus,
                                                  model_name,
                                                  lr,
                                                  train_batch_size,
                                                  eval_batch_size,
                                                  epochs_per_corpus)


if __name__ == "__main__":
    train_directory = "/home/zakiamin/PycharmProjects/hpo_translation/corpus/train/"
    token = "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"
    seed = 17
    torch.manual_seed(seed)
    epochs_per_corpus, lr, batch_size = 3, 1e-6, 8
    main(token, train_directory, epochs_per_corpus, lr, batch_size, batch_size * 2)
