from enum import Enum

from evaluate import load
from sentence_transformers import SentenceTransformer, util
from torchmetrics.text import TranslationEditRate

# Download the model before the first run
# nltk.download('punkt')
similarity_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

lowercase = True
sacrebleu = load("sacrebleu")
ter = TranslationEditRate(return_sentence_level_score=True, lowercase=lowercase, normalize=True, no_punctuation=True)


class SentenceSimilarity(Enum):
    """Enum for similarity metrics between two sentences.
    Each metric must implement the evaluate method."""
    SIMPLE = 0
    SACREBLEU = 1
    TER = 2
    SEMANTIC_SIMILARITY = 3

    def evaluate(self, reference: str, candidate: str) -> float:
        """Evaluate the given string similarity metric between two strings.
        Performs simple string cleaning for whitespace and punctuation.
        :param reference: reference and official term
        :param candidate: model-produced translated term
        :return: similarity score when evaluating this specific metric
        """
        match self:
            case SentenceSimilarity.SIMPLE:
                return 1 if reference == candidate else 0

            case SentenceSimilarity.SACREBLEU:
                references = [[reference]]
                predictions = [candidate]
                results = sacrebleu.compute(predictions=predictions,
                                            references=references,
                                            use_effective_order=True,
                                            tokenize="intl",
                                            lowercase=lowercase)
                return round(results["score"], 1)

            case SentenceSimilarity.TER:
                score = ter(reference, candidate)[0].item()
                return round(score, 1)

            case SentenceSimilarity.SEMANTIC_SIMILARITY:
                reference, candidate = reference.lower(), candidate.lower()
                query_embedding = similarity_model.encode(reference)
                passage_embedding = similarity_model.encode(candidate)
                score = util.cos_sim(query_embedding, passage_embedding)[0].item()
                return round(score * 100, 1)

    def __str__(self):
        match self:
            case SentenceSimilarity.SIMPLE:
                return "simple"
            case SentenceSimilarity.SACREBLEU:
                return "sacrebleu"
            case SentenceSimilarity.TER:
                return "'ter'"
            case SentenceSimilarity.SEMANTIC_SIMILARITY:
                return "semsim"


if __name__ == "__main__":
    reference = "Le harán la prueba para el Zika durante el embarazo."
    prediction = "Se le hará una prueba para detectar el virus del Zika durante el embarazo."
    for metric in SentenceSimilarity:
        print(f"{metric}: {metric.evaluate(reference, prediction)}")
