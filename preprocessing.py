import string
import json
import nltk
import sys
import re
import os
import numpy

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.feature import Word2Vec
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize

CONTRACTION_MAP = {

    "ain't": "is not",
    "aren't": "are not",
    "arent": "are not",
    "btw": "by the way",
    "can't": "can not",
    "cant": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "cause": "because",
    "could've": "could have",
    "couldve": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "thats": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "theres": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "whats": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"

}

stop_words = ['to', 'and', 'or', 'be', 'in', 'of', 'at', 'a', 'an', 'the', 'i', 'we', 'you', 'they', 'he', 'she', 'it',
              'that', 'this', 'those', 'these', 'by', 'my', 'your', 'our', 'his', 'her', 'its','their']


def preprocess(data):
    # remove urls
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)

    # lower case
    text = text.lower()

    # tokenization
    # tmp = word_tokenize(text)
    # print("Word_tokenize")
    # print(tmp)

    tknz = TweetTokenizer()
    text = tknz.tokenize(text)


    # replace contractions
    text = ' '.join([replace_contraction(word=word) for word in text])

    # remove digits
    text = re.sub(r'[' + string.digits + ']+', '', text)

    text = re.sub(r'\b\w{1,3}\b', '', text)

    # remove puntaction
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)

    # remove whitespaces
    text = re.sub(r'[' + string.whitespace + ']+', ' ', text)

    # tokenize
    text = word_tokenize(text)
    # print("TweetTokenizer 2 ")

    # review text vuoto--> eliminare
    # parallelizzare anche parole?

    # lemmatizzation
    lemmatizer = WordNetLemmatizer()

    text = [word for word in text if word not in stop_words]

    # verbs
    text = [lemmatizer.lemmatize(word, pos='v') for word in text]

    # noun
    text = [lemmatizer.lemmatize(word, pos='n') for word in text]

    # adjective
    text = [lemmatizer.lemmatize(word, pos='a') for word in text]

    # adverb
    text = [lemmatizer.lemmatize(word, pos='r') for word in text]

    return text


def replace_contraction(word):
    ret = word

    print()
    print(word)
    print("====")
    if word in CONTRACTION_MAP.keys():
        ret = CONTRACTION_MAP[word]
        print(word)
        print(ret)
    print("====")
    print()

    return ret


def read_dataset(sc, dataset, trainSplit, testSplit):
    data = sc.textFile(dataset)

    (train, test) = data.randomSplit([trainSplit, testSplit])

    return train.map(lambda x: json.loads(x)) \
        .filter(lambda x: "reviewText" in x and "overall" in x) \
        .map(lambda x: preprocess(x["reviewText"])), test


def getOriginalSentiment(sc, dataset):
    data = sc.textFile(dataset)
    rdd = data.map(lambda x: json.loads(x))

    sentiment = rdd.map(lambda y: map_overall_to_sentiment(y["overall"]))
    return sentiment


def map_overall_to_sentiment(points):
    if points <= 2:
        return -1
    elif points == 3:
        return 0
    else:
        return 1