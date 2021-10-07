import string
import json
import nltk
from nltk.corpus import brown
import sys
import re
import os
import numpy
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from word2vec import getWord2Vec
from pyspark.mllib.feature import Word2Vec, Word2VecModel
from gensim.models import KeyedVectors
import gensim.downloader as api

from preprocessing import preprocess, read_dataset, getOriginalSentiment
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import os

os.environ['SPARK_HOME'] = 'C:\Spark\spark-3.0.3-bin-hadoop2.7'

dataset = 'C:\\Users\\Alessia\\PycharmProjects\\BigData\\dataset\\Kindle.json'
word2vecPath = 'C:\\Users\\Alessia\\PycharmProjects\\BigData\\models\\GoogleNews-vectors-negative300.bin'


def init_spark():
    spark = SparkSession.builder.appName("BigData").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


if __name__ == '__main__':
    spark, sc = init_spark()
    #model = Word2VecModel.load(sc, "C:\\Users\\Alessia\\PycharmProjects\\BigData\\models"
    #                               "\\ModelWithEverythingButGutenberg.model")
    # data = getWord2Vec(sc)
    reviews = read_dataset(sc, dataset)

    word2vec = Word2Vec().setVectorSize(300).setMinCount(100)
    model = word2vec.fit(reviews)
    model.save(sc, 'C:\\Users\\Alessia\\PycharmProjects\\BigData\\models\\ModelWith50Kindle.model')
    sentiment = getOriginalSentiment(sc, dataset)

    #rev = model.transform(reviews)
    #print(sentiment.collect())

    print(list(model.findSynonyms("good", 10)))
    print(list(model.findSynonyms("book", 10)))
    print(list(model.findSynonyms("perfect", 10)))

    # loadedWord2Vec = Word2VecModel.load(sc, word2vecPath)

    # models = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
    # positive_cluster_center = models.cluster_centers_[0]
    # negative_cluster_center = models.cluster_centers_[1]
