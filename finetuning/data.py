from datasets import load_dataset, DatasetDict


def load_corpus(data_filename: str, validation_proportion: float, seed: int) -> DatasetDict:
    """Reads a JSONL file for a parallel corpus
    :param data_filename: path to the JSONL file containing the parallel corpus
    :param validation_proportion: the proportion of the data to use for validation
    :param seed: the random seed to use for splitting the data"""
    training_data = load_dataset("json", data_files=data_filename)
    train_proportion = 1 - validation_proportion
    split_datasets = training_data["train"].train_test_split(train_size=train_proportion, seed=seed)
    split_datasets["validation"] = split_datasets.pop("test")
    return split_datasets
