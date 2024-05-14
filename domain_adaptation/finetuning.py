from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq, MarianTokenizer, MarianMTModel


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


class FineTuning:
    def __init__(self, checkpoint_name: str, max_length: int):
        self.checkpoint_name = checkpoint_name
        self.max_length = max_length
        self.model = MarianMTModel.from_pretrained(checkpoint_name)
        self.tokenizer = MarianTokenizer.from_pretrained(checkpoint_name)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    model=self.model,
                                                    max_length=max_length)

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
