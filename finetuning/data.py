from datasets import load_dataset, DatasetDict
from transformers import MarianTokenizer


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


def marian_tokenizer() -> MarianTokenizer:
    """Returns the original MarianTokenizer for Helsinki-NLP/opus-mt-en-es"""
    checkpoint_name = "Helsinki-NLP/opus-mt-en-es"
    return MarianTokenizer.from_pretrained(checkpoint_name)


def preprocess_with_tokens(tokenizer: MarianTokenizer, examples, max_length: int):
    """Preprocesses the data for fine-tuning using the MarianTokenizer"""
    inputs = [ex for ex in examples["en"]]
    targets = [ex for ex in examples["es"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def tokenize_all_datasets(tokenizer: MarianTokenizer, data: DatasetDict, max_length: int) -> DatasetDict:
    """Applies tokenization pre-processing to all datasets"""
    return data.map(
        lambda examples: preprocess_with_tokens(tokenizer, examples, max_length),
        batched=True,
    )
