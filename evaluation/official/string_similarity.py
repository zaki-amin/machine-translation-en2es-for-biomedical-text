from enum import Enum

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')


class SimilarityMetric(Enum):
    """Enum for string similarity metrics."""
    BLEU = 0
    SIMPLE_MATCH = 1

    def evaluate(self, reference: str, candidate: str) -> float:
        """Evaluate the given string similarity metric between two strings."""
        match self:
            case SimilarityMetric.BLEU:
                reference_tokens = nltk.word_tokenize(reference.lower())
                candidate_tokens = nltk.word_tokenize(candidate.lower())
                return sentence_bleu([reference_tokens], candidate_tokens,
                                     smoothing_function=SmoothingFunction().method1)

            case SimilarityMetric.SIMPLE_MATCH:
                return 1 if reference == candidate else 0


if __name__ == "__main__":
    metric = SimilarityMetric.BLEU
    print(metric.evaluate("Frecuencia", "Frecuencia"))
