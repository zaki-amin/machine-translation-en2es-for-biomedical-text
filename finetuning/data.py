from datasets import load_dataset, DatasetDict


def load_corpus(data_filename: str) -> DatasetDict:
    """Reads a JSONL file for a parallel corpus
    :param data_filename: path to the JSONL file containing the parallel corpus"""
    training_data = load_dataset("json", data_files=data_filename)
    return training_data["train"]
