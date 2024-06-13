from enum import Enum

from evaluate import load
from sentence_transformers import SentenceTransformer, util
from torchmetrics.text import TranslationEditRate


# Download the model before the first run
# nltk.download('punkt')


class CorpusSimilarity(Enum):
    """Enum for string similarity metrics. Each metric must implement the evaluate method."""
    SACREBLEU = 0
    TER = 1
    SEMANTIC_SIMILARITY = 2

    def evaluate(self, references: list[str], predictions: list[str]) -> float:
        """Evaluate the given similarity metric between two corpora.
        :param references: list of references (official translations)
        :param predictions: list of candidates (model translations)
        :return: corpus similarity percentage when evaluating this specific metric
        """
        match self:
            case CorpusSimilarity.SACREBLEU:
                # SacreBLEU expects a list of references for each candidate
                references = [[ref] for ref in references]
                predictions = [cand for cand in predictions]
                sacrebleu = load("sacrebleu")
                results = sacrebleu.compute(predictions=predictions, references=references, tokenize='intl')
                return round(results["score"], 1)

            case CorpusSimilarity.TER:
                references = [[ref] for ref in references]
                predictions = [cand for cand in predictions]
                ter = TranslationEditRate(return_sentence_level_score=False, lowercase=False, normalize=True)
                inverted_score = 1 - ter(predictions, references).item()
                return round(inverted_score * 100, 1)

            case CorpusSimilarity.SEMANTIC_SIMILARITY:
                similarity_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
                n = len(references)
                score = 0
                for reference, candidate in zip(references, predictions):
                    reference_embedding = similarity_model.encode(reference)
                    candidate_embedding = similarity_model.encode(candidate)
                    cosine_similarity = util.cos_sim(reference_embedding, candidate_embedding)
                    score += cosine_similarity[0].item()
                return round(score / n * 100, 1)

    def __str__(self):
        match self:
            case CorpusSimilarity.SACREBLEU:
                return "sacrebleu"
            case CorpusSimilarity.TER:
                return "`ter`"
            case CorpusSimilarity.SEMANTIC_SIMILARITY:
                return "semsim"
