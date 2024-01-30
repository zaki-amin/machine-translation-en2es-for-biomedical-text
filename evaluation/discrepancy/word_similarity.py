import spacy

from evaluation.utility.text_functions import word_split, trim_string

nlp = spacy.load("en_core_web_md")


def word_differences(official_term: str, model_term: str) -> tuple[set[str], set[str]]:
    """Returns the words that are in the official term but not in the model term and vice versa"""
    official_words = word_split(official_term)
    model_words = word_split(model_term)
    # Clean all the words so any attached punctuation does not affect the comparison
    official_words = {trim_string(word) for word in official_words}
    model_words = {trim_string(word) for word in model_words}
    return official_words - model_words, model_words - official_words


def most_similar_word(word: str, target_sentence: str) -> tuple[str, float]:
    """
    :param word: The word to compare
    :param target_sentence: The target sentence to compare against
    :return: A pair of the most similar word in the target sentence and the similarity score from spacy"""
    document = nlp(target_sentence)
    word_representation = nlp(word)

    similarity_scores = {token.text: token.similarity(word_representation) for token in document}
    max_key = max(similarity_scores, key=lambda k: similarity_scores[k])
    max_value = similarity_scores[max_key]
    return max_key, max_value


def all_word_similarities(mistranslated: str, target_sentence: str) -> dict[str, tuple[str, float]]:
    """
    :param mistranslated: The mistranslated sentence
    :param target_sentence: The target translation
    :return: A list of the most similar word in the target sentence for each word in the mistranslated sentence"""
    all_similarities = {}
    for word in mistranslated.split():
        all_similarities[word] = most_similar_word(word, target_sentence)
    return all_similarities


if __name__ == "__main__":
    print(most_similar_word("chips", "I like salty fries and hamburgers."))
    print(all_word_similarities("love sandwiches chips", "I like salty fries and hamburgers."))
