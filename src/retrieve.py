import pickle as pkl
import string
import numpy as np
# from preprocess import Audio_Processor
import nltk
from nltk.corpus import wordnet

SR = 16000
blocksize = int(SR/2)
overlap = int(SR/4)

wn = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

def clean_input(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = text.split()
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    lemmas = set()
    for token in text:
        lemmas += set([synset.lemmas()[0].name() for synset in wordnet.synsets(token)])
    return lemmas

if __name__ == "__main__":
    while 1:
        query = input("What do you want to find?: ")

        num = input("Number of Documents to Return: ")

        actual_query = clean_input(query)
        print(actual_query)