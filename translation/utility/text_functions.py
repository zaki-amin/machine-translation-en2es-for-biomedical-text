import string


def word_split(phrase: str) -> set[str]:
    """Splits a phrase into a set of its constituent words"""
    return set(phrase.split())


def trim_string(sentence: str) -> str:
    """Cleans punctuation and whitespace from a sentence"""
    return sentence.strip(string.punctuation).strip()
