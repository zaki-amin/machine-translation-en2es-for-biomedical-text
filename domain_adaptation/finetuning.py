import evaluate
import huggingface_hub
import numpy as np
import torch
from accelerate import Accelerator
from datasets import DatasetDict
from huggingface_hub import Repository, get_full_repo_name
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq, MarianTokenizer, MarianMTModel, AdamW, get_scheduler, GenerationConfig

from domain_adaptation.corpus import load_all_corpora


class FineTuning:
    def __init__(self, checkpoint_name: str):
        self.checkpoint_name = checkpoint_name
        self.generation_config = generation_config()
        self.device = cuda_if_possible()
        self.model = MarianMTModel.from_pretrained(checkpoint_name, device_map=self.device)
        self.tokenizer = MarianTokenizer.from_pretrained(checkpoint_name)
        # Using PyTorch hence 'pt'
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                                    model=self.model,
                                                    max_length=self.generation_config.max_length,
                                                    return_tensors="pt")
        self.metric = evaluate.load("sacrebleu")

    def preprocess_with_tokens(self, examples):
        """Preprocesses the data for fine-tuning using the MarianTokenizer"""
        return self.tokenizer(
            examples["en"],
            text_target=examples["es"],
            max_length=self.generation_config.max_length,
            padding="max_length",
            truncation=True,
        )

    def tokenize_all_datasets(self, data: DatasetDict) -> DatasetDict:
        """Applies tokenization pre-processing to all datasets"""
        tokenized = data.map(lambda examples: self.preprocess_with_tokens(examples),
                             batched=True,
                             remove_columns=data["train"].column_names)
        tokenized.set_format("torch")
        return tokenized

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

    def save_and_upload(self, accelerator, epoch, model, output_dir, repo):
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        self.generation_config.save_pretrained(output_dir, push_to_hub=True)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            self.tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )


def generation_config() -> GenerationConfig:
    gen_config = GenerationConfig()
    gen_config.max_length = 512
    gen_config.num_beams = 4
    gen_config.bad_words_ids = [[65000]]
    gen_config.forced_eos_token_id = 0

    gen_config.save_pretrained('./generation_config')
    return gen_config


def login_and_get_repo(hf_token: str):
    huggingface_hub.login(token=hf_token)
    model_name = "helsinki-biomedical-finetuned"
    repo_name = get_full_repo_name(model_name)
    repo = Repository(model_name, repo_name)
    repo.git_pull()
    return model_name, repo


def cuda_if_possible() -> str:
    """If a GPU is available, empties cache and returns 'cuda', otherwise 'cpu'"""
    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
    print(f"Using device: {device}")
    return device
