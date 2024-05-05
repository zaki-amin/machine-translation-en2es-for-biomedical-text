from enum import Enum

import jiwer
import nltk
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

# nltk.download('punkt')
similarity_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


# This class is intended to evaluate between just two strings, not a list of strings.


class SimilarityMetric(Enum):
    """Enum for string similarity metrics. Each metric must implement the evaluate method."""
    SIMPLE = 0
    BLEU = 1
    SACREBLEU = 1
    WER = 2
    SEMANTIC_SIMILARITY = 3

    def evaluate(self, reference: str, candidate: str) -> float:
        """Evaluate the given string similarity metric between two strings.
        Performs simple string cleaning for whitespace and punctuation.
        :param reference: reference and official term
        :param candidate: model-produced translated term
        :return: similarity score when evaluating this specific metric
        """

        match self:
            case SimilarityMetric.SIMPLE:
                return 1 if reference == candidate else 0

            case SimilarityMetric.BLEU:
                reference_tokens = nltk.word_tokenize(reference.lower())
                candidate_tokens = nltk.word_tokenize(candidate.lower())
                return sentence_bleu([reference_tokens], candidate_tokens,
                                     smoothing_function=SmoothingFunction().method1)

            case SimilarityMetric.SACREBLEU:
                bleu = sacrebleu.raw_corpus_bleu(candidate, [reference])
                return bleu.score

            case SimilarityMetric.WER:
                return jiwer.wer(reference, candidate)

            case SimilarityMetric.SEMANTIC_SIMILARITY:
                query_embedding = similarity_model.encode(reference)
                passage_embedding = similarity_model.encode(candidate)
                cosine_similarity = util.cos_sim(query_embedding, passage_embedding)
                return cosine_similarity[0].item()


if __name__ == "__main__":
    metric = SimilarityMetric.SACREBLEU
    print(metric.evaluate("Frecuencia diario", "Frecuencia diario"))
