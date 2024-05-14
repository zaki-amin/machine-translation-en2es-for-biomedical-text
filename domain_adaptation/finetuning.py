import evaluate
import huggingface_hub
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, MarianTokenizer, MarianMTModel, AdamWeightDecay, get_scheduler


class FineTuning:
    def __init__(self, checkpoint_name: str, max_length: int):
        self.checkpoint_name = checkpoint_name
        self.max_length = max_length
        self.model = MarianMTModel.from_pretrained(checkpoint_name)
        self.tokenizer = MarianTokenizer.from_pretrained(checkpoint_name)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    model=self.model,
                                                    max_length=max_length,
                                                    return_tensors="tf")
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

    def finetune_model(self, tokenized_texts: DatasetDict, batch_size=16):
        """Fine-tunes the model on the tokenized texts using the specified batch size"""
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
        optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        self.model.compile(optimizer)

        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, validation_dataloader
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=128,
                    )
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(generated_tokens)
                labels_gathered = accelerator.gather(labels)

                decoded_preds, decoded_labels = self.postprocess(predictions_gathered, labels_gathered)
                self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            results = self.metric.compute()
            print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            output_dir = "finetuned"
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)


def load_corpus(data_filename: str, validation_proportion: float, seed: int) -> DatasetDict:
    """Reads a JSONL file for a parallel corpus into a DatasetDict for fine-tuning
    :param data_filename: path to the JSONL file containing the parallel corpus
    :param validation_proportion: the proportion of the data to use for validation
    :param seed: the random seed to use for splitting the data"""
    training_data = load_dataset("json", data_files=data_filename)
    train_proportion = 1 - validation_proportion
    split_datasets = training_data["train"].train_test_split(train_size=train_proportion, seed=seed)
    split_datasets["validation"] = split_datasets.pop("test")
    return split_datasets


def main(hf_token: str):
    huggingface_hub.login(token=hf_token)
    filepath = "../corpus/train/medline.jsonl"
    biomedical_texts = load_corpus(filepath, 0.2, 42)
    fine_tuning = FineTuning("Helsinki-NLP/opus-mt-en-es", 512)
    tokenized_texts = fine_tuning.tokenize_all_datasets(biomedical_texts)
    fine_tuning.finetune_model(tokenized_texts)


if __name__ == "__main__":
    hf_token = "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"
    main(hf_token)
