import numpy as np
import torch
from datasets import DatasetDict
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AdamW, get_scheduler

from domain_adaptation.corpus import load_all_corpora
from domain_adaptation.finetuning import FineTuning, login_and_get_repo


class FineTuningTrainer(FineTuning):
    def __init__(self, checkpoint_name: str):
        super().__init__(checkpoint_name)

    def finetune_with_trainer(self,
                              corpora: DatasetDict,
                              model_name: str,
                              learning_rate: float,
                              train_batch_size: int,
                              eval_batch_size: int,
                              training_epochs: int):
        tokenized_texts = self.tokenize_all_datasets(corpora)
        args = Seq2SeqTrainingArguments(
            model_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            lr_scheduler_type="cosine",
            metric_for_best_model="eval_loss",
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=training_epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True,
            gradient_accumulation_steps=4,
        )
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=1,
        )
        trainer = Seq2SeqTrainer(
            self.model.to(self.device),
            args,
            train_dataset=tokenized_texts["train"],
            eval_dataset=tokenized_texts["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, scheduler)
        )

        initial_results = trainer.evaluate(max_length=self.generation_config.max_length)
        print(f"Initial eval_bleu: {initial_results['eval_bleu']}")
        trainer.train()
        final_results = trainer.evaluate(max_length=self.generation_config.max_length)
        print(f"Final eval_bleu: {final_results['eval_bleu']}")
        trainer.push_to_hub(tags="translation", commit_message="Training complete")

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}


def main(hf_token: str,
         train_filepath: str,
         training_epochs: int,
         learning_rate: float,
         train_batch_size: int,
         eval_batch_size: int):
    model_name, repo = login_and_get_repo(hf_token)
    biomedical_corpora = load_all_corpora(train_filepath, 0.1, 42)
    trainer_fine_tuning = FineTuningTrainer("Helsinki-NLP/opus-mt-en-es")
    trainer_fine_tuning.finetune_with_trainer(biomedical_corpora,
                                              model_name,
                                              learning_rate,
                                              train_batch_size,
                                              eval_batch_size,
                                              training_epochs)


if __name__ == "__main__":
    train_directory = "../corpus/train/"
    token = input("Enter Hugging Face API token: ")
    seed = 17
    torch.manual_seed(seed)
    epochs, lr, batch_size = 25, 1e-5, 8
    main(token, train_directory, epochs, lr, batch_size, batch_size * 2)
