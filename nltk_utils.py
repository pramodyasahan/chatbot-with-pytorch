import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(sentence):
    return PorterStemmer().stem(sentence.lower())


def bag_of_words(sentence, words):
    sentence_words = [stem(w) for w in sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        if w in sentence_words:
            bag[i] = 1.0

    return bag