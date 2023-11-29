import string
from enum import Enum

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util

import sacrebleu

# nltk.download('punkt')
similarity_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


class SimilarityMetric(Enum):
    """Enum for string similarity metrics."""
    BLEU = 0
    SIMPLE = 1
    EDIT_DISTANCE = 2
    SEMANTIC_SIMILARITY = 3
    SACREBLEU = 4

    def evaluate(self, reference: str, candidate: str) -> float:
        """Evaluate the given string similarity metric between two strings.
        Performs simple string cleaning for whitespace and punctuation.
        """
        reference = reference.strip(string.punctuation).strip()
        candidate = candidate.strip(string.punctuation).strip()

        match self:
            case SimilarityMetric.BLEU:
                reference_tokens = nltk.word_tokenize(reference.lower())
                candidate_tokens = nltk.word_tokenize(candidate.lower())
                return sentence_bleu([reference_tokens], candidate_tokens,
                                     smoothing_function=SmoothingFunction().method1)

            case SimilarityMetric.SIMPLE:
                return 1 if reference == candidate else 0

            case SimilarityMetric.EDIT_DISTANCE:
                return 1 - nltk.edit_distance(reference, candidate) / max(len(reference), len(candidate))

            case SimilarityMetric.SEMANTIC_SIMILARITY:
                query_embedding = similarity_model.encode(reference)
                passage_embedding = similarity_model.encode(candidate)
                cosine_similarity = util.cos_sim(query_embedding, passage_embedding)
                return cosine_similarity[0].item()

            case SimilarityMetric.SACREBLEU:
                bleu = sacrebleu.raw_corpus_bleu(candidate, [reference])
                return bleu.score


if __name__ == "__main__":
    metric = SimilarityMetric.SACREBLEU
    print(metric.evaluate("Frecuencia diario", "Frecuencia diario"))
