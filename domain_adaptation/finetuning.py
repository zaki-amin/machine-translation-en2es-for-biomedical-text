import evaluate
import huggingface_hub
import numpy as np
import torch

from accelerate import Accelerator
from datasets import DatasetDict
from huggingface_hub import Repository, get_full_repo_name
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, MarianTokenizer, MarianMTModel, AdamW, get_scheduler

from domain_adaptation.corpus import load_all_corpora


class FineTuning:
    def __init__(self, checkpoint_name: str, max_length: int):
        self.checkpoint_name = checkpoint_name
        self.max_length = max_length
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MarianMTModel.from_pretrained(checkpoint_name, device_map=device)
        self.tokenizer = MarianTokenizer.from_pretrained(checkpoint_name)
        # Using PyTorch hence 'pt'
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    model=self.model,
                                                    max_length=max_length,
                                                    return_tensors="pt")
        self.metric = evaluate.load("sacrebleu")

    def preprocess_with_tokens(self, examples):
        """Preprocesses the data for fine-tuning using the MarianTokenizer"""
        return self.tokenizer(
            examples["en"],
            text_target=examples["es"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

    def tokenize_all_datasets(self, data: DatasetDict) -> DatasetDict:
        """Applies tokenization pre-processing to all datasets"""
        return data.map(
            lambda examples: self.preprocess_with_tokens(examples),
            batched=True,
            remove_columns=data["train"].column_names,
        )

    def postprocess(self, predictions: torch.Tensor, labels: torch.Tensor):
        predictions = predictions.cpu().numpy()
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_predictions = [pred.strip() for pred in decoded_predictions]

        labels = labels.cpu().numpy()
        # Replace -100 in the labels because they cannot be decoded
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [[label.strip()] for label in decoded_labels]

        return decoded_predictions, decoded_labels

    def finetune_model(self,
                       tokenized_texts: DatasetDict,
                       train_epochs,
                       batch_size,
                       output_dir: str,
                       repo: Repository):
        """Fine-tunes the model
        :param tokenized_texts: the tokenized datasets to use for fine-tuning
        :param train_epochs: the number of epochs to train for
        :param batch_size: the batch size to use for training and evaluation
        """
        tokenized_texts.set_format("torch")
        train_dataloader = DataLoader(
            tokenized_texts["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=batch_size,
        )
        validation_dataloader = DataLoader(
            tokenized_texts["validation"],
            collate_fn=self.data_collator,
            batch_size=batch_size,
        )
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, validation_dataloader
        )

        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "reduce_lr_on_plateau",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(train_epochs):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            total_eval_loss = 0
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_eval_loss += loss.item()

                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=self.max_length,
                    )

                labels = batch["labels"]
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                predictions_gathered = accelerator.gather(generated_tokens)
                labels_gathered = accelerator.gather(labels)
                decoded_preds, decoded_labels = self.postprocess(predictions_gathered, labels_gathered)
                self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            avg_val_loss = total_eval_loss / len(eval_dataloader)
            lr_scheduler.step(avg_val_loss)

            results = self.metric.compute()
            print(f"Epoch {epoch + 1}, BLEU score: {results['score']:.2f}")

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False
                )


def main(hf_token: str):
    huggingface_hub.login(token=hf_token)
    model_name = "helsinki-biomedical-finetuned"
    repo_name = get_full_repo_name(model_name)
    repo = Repository(model_name, repo_name)

    biomedical_corpora = load_all_corpora("smalldata/", 0.2, 42)

    fine_tuning = FineTuning("Helsinki-NLP/opus-mt-en-es", 512)
    tokenized_texts = fine_tuning.tokenize_all_datasets(biomedical_corpora)
    fine_tuning.finetune_model(tokenized_texts, 3, 16, model_name, repo)


if __name__ == "__main__":
    token = "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"
    main(token)
