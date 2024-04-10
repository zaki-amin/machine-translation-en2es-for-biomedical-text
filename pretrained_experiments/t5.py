import evaluate
import huggingface_hub
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, \
    AdamWeightDecay, TFAutoModelForSeq2SeqLM, KerasMetricCallback, pipeline, PushToHubCallback

# Settings for the model and Hugging Face
model_checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = evaluate.load("sacrebleu")
hf_token = "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"


def load_data(data_filename: str, seed: int = 69) -> DatasetDict:
    """Reads a JSONL file for a parallel corpus and splits it into training and test sets
    :param data_filename: path to the JSONL file containing the parallel corpus
    :param seed: seed for splitting of data into training and test sets
    :return: training and test sets for the parallel corpus"""
    training_data = load_dataset("json", data_files=data_filename)
    return training_data["train"].train_test_split(train_size=0.9, seed=seed)


def tokenize_datasets(biomedical_texts: DatasetDict, max_length: int = 256) -> DatasetDict:
    """Tokenizes the datasets in preparation for training an ES-EN biomedical text translation model
    :param biomedical_texts: dataset split into training and test sets
    :param max_length: maximum length of the tokenized sequences in the model
    :return: tokenized datasets for training and testing the model"""

    def preprocess_function(examples: DatasetDict):
        prefix = "translate Spanish to English: "
        inputs = [prefix + ex for ex in examples['es']]
        targets = [ex for ex in examples['en']]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs

    tokenized_texts = biomedical_texts.map(
        preprocess_function,
        batched=True
    )
    return tokenized_texts


def postprocess_text(predictions: torch.Tensor, labels: torch.Tensor) -> tuple[list[str], list[list[str]]]:
    """Cleans the predictions and labels for evaluation
    :param predictions: tensor of predicted translations
    :param labels: tensor of actual translations
    :return: cleaned predictions and labels for evaluation"""
    preds = [pred.strip() for pred in predictions]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    """Computes SacreBLEU score and generation length for the model
    :param eval_preds: predictions and labels for evaluation
    :return: BLEU score and generation length for the model"""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def finetune_model(tokenized_texts: DatasetDict):
    """Fine-tunes the T5 model on the biomedical text translation dataset, saving it to Hugging Face
    :param tokenized_texts: tokenized datasets for training and testing the model"""
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint, return_tensors="tf")
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    tf_train_set = model.prepare_tf_dataset(
        tokenized_texts["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )
    tf_test_set = model.prepare_tf_dataset(
        tokenized_texts["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
    push_to_hub_callback = PushToHubCallback(
        output_dir="biomedical_text_model",
        tokenizer=tokenizer,
    )
    callbacks = [metric_callback, push_to_hub_callback]
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)


def main():
    huggingface_hub.login(token=hf_token)

    biomedical_texts = load_data("small_translations.jsonl")
    tokenized_texts = tokenize_datasets(biomedical_texts)
    finetune_model(tokenized_texts)

    text = "translate Spanish to English: Sanidad militar: revista de sanidad de las Fuerzas Armadas de España"
    translator = pipeline("translation", model="biomedical_text_model")
    print(translator(text))


if __name__ == "__main__":
    main()
