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


def init_spark():

    spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():

    #spark, sc = init_spark()
    #nums = sc.parallelize([1, 2, 3, 4])
    #print(nums.map(lambda x: x * x).collect())

    spark, sc = init_spark()
    dataset = sc.textFile(name='dataset/prova.json')
    # dataset = sc.textFile(name='C:\\Users\\Alessia\\PycharmProjects\\BigData\\dataset\\Appliances_5.json')
    # dataset = sc.textFile(name='C:\\Users\\Alessia\\PycharmProjects\\BigData\\dataset\\prova.json')
    reviews = dataset.map(lambda x: json.loads(x)).map(lambda x: clean_text(x['reviewText']))
    print(reviews.collect())
    #print(dataset.map(lambda x: json.loads(x)).map(lambda x: (clean_text(x['reviewText']), x['overall'])).collect())
    #print(reviews.collect())
    #Word2vec = Word2Vec().setMinCount(10).setVectorSize(100)
    #model = word2vec.fit(reviews)
    #synonyms = model.findSynonyms('well', 10)

    #for word, cosine_distance in synonyms:
        #print("{}: {}".format(word, cosine_distance))


def clean_text(text):

    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = text.lower()
    tmp = word_tokenize(text)
    tknz = TweetTokenizer()
    text = tknz.tokenize(text)

    text = ' '.join([replace_contraction(word=word) for word in text])

    text = re.sub(r'[' + string.digits + ']+', '', text)
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)
    text = re.sub(r'[' + string.whitespace + ']+', ' ', text)
    text = word_tokenize(text)
    #review text vuoto--> eliminare
    #parallelizzare anche parole?
    lemmatizer = WordNetLemmatizer()

    from nltk.corpus import stopwords, wordnet

    text = [word for word in text if word not in stopwords.words('english')]

    text = [lemmatizer.lemmatize(word, pos='v') for word in text]
    text = [lemmatizer.lemmatize(word, pos='n') for word in text]
    text = [lemmatizer.lemmatize(word, pos='a') for word in text]
    text = [lemmatizer.lemmatize(word, pos='r') for word in text]
    return text


def replace_contraction(word):

    ret = word

    if word in CONTRACTION_MAP.keys():

        ret = CONTRACTION_MAP[word]

    return ret


if __name__ == '__main__':
    #nltk.download("all")
    main()

