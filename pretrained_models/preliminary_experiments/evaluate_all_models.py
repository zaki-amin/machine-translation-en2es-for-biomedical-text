import json
import pprint

import pandas as pd
from tqdm import tqdm

from evaluation.official.string_similarity import SimilarityMetric
from pretrained_models.preliminary_experiments.translation_model import TranslationModel, NLLBModel


def load_sentences(test_dataset: str) -> pd.DataFrame:
    """Loads a test dataset in .jsonl format into a dataframe
    :param test_dataset: filename of the test dataset
    :return pd.Dataframe: the parallel corpus as a dataframe"""
    data = []
    with open(test_dataset, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def evaluate_models_on(mt_models: list[TranslationModel], test_sentences: pd.DataFrame) -> dict[
    TranslationModel, dict[SimilarityMetric, float]]:
    """For each model, obtains the average for all metrics over all test sentences.
    :param mt_models: the list of translation models to evaluate
    :param test_sentences: a dataframe for a parallel corpus"""

    results = {model: {metric: 0 for metric in SimilarityMetric} for model in mt_models}

    for model in mt_models:
        print(f"Model: {model}")
        for _, row in tqdm(test_sentences.iterrows()):
            english, spanish = row['en'], row['es']
            for metric in SimilarityMetric:
                similarity = metric.evaluate(spanish, model.translate(english))
                results[model][metric] += similarity

    n = test_sentences.shape[0]
    for model in mt_models:
        for metric in results[model]:
            results[model][metric] /= n

    return results


def evaluate_on_all_test_data(translation_models: list[TranslationModel], test_datasets: list[str]):
    for test_dataset in test_datasets:
        print(f"Test dataset: {test_dataset}")
        test_sentences = load_sentences(test_dataset)
        model_metrics = evaluate_models_on(translation_models, test_sentences)
        pprint.pprint(model_metrics)


if __name__ == "__main__":
    directory_prefix = "/Users/zaki/PycharmProjects/hpo_translation/corpus/test/"
    filenames = ["abstract5.jsonl"] # + ["abstracts.jsonl", "clinspen.jsonl", "khresmoi.jsonl"]
    all_test_datasets = [directory_prefix + filename for filename in filenames]
    
    all_models = [NLLBModel()]

    evaluate_on_all_test_data(all_models, all_test_datasets)
