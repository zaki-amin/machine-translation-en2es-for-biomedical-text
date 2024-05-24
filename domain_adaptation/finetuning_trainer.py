from datasets import DatasetDict
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

from domain_adaptation.corpus import load_all_corpora
from domain_adaptation.finetuning import FineTuning, login_and_get_repo


class FineTuningTrainer(FineTuning):
    def __init__(self, checkpoint_name: str):
        super().__init__(checkpoint_name)

    def finetune_with_trainer(self,
                              corpora: DatasetDict,
                              model_name: str,
                              lr: float,
                              train_batch_size: int,
                              eval_batch_size: int,
                              epochs: int):
        tokenized_texts = self.tokenize_all_datasets(corpora)
        args = Seq2SeqTrainingArguments(
            model_name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True,
            gradient_accumulation_steps=2,
        )
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_texts["train"],
            eval_dataset=tokenized_texts["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
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
         epochs: int,
         lr: float,
         train_batch_size: int,
         eval_batch_size: int):
    model_name, repo = login_and_get_repo(hf_token)
    biomedical_corpora = load_all_corpora(train_filepath, 0.1, 42)
    trainer_fine_tuning = FineTuningTrainer("Helsinki-NLP/opus-mt-en-es")
    trainer_fine_tuning.finetune_with_trainer(biomedical_corpora,
                                              model_name,
                                              lr,
                                              train_batch_size,
                                              eval_batch_size,
                                              epochs)


if __name__ == "__main__":
    # train_directory = "smalldata/"
    train_directory = "/home/zakiamin/PycharmProjects/hpo_translation/corpus/train/"
    token = "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"
    epochs, lr, batch_size = 15, 5e-7, 8
    main(token, train_directory, epochs, lr, batch_size, batch_size * 2)
