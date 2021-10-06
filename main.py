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
from preprocessing import preprocess, read_dataset
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import os

os.environ['SPARK_HOME'] = 'C:\Spark\spark-3.0.3-bin-hadoop2.7'

dataset = 'C:\\Users\\Alessia\\PycharmProjects\\BigDataProgetto\\dataset\\test.json'


def init_spark():
    spark = SparkSession.builder.appName("BigData").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


if __name__ == '__main__':
    spark, sc = init_spark()
    reviews = read_dataset(sc, dataset)
    # print(reviews)
    model = Word2Vec().setMinCount(5).setVectorSize(500).setSeed(42).fit(reviews)
    # print(model.getVectors())
    print(list(model.findSynonyms("good", 10)))
    # model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
    # positive_cluster_center = model.cluster_centers_[0]
    # negative_cluster_center = model.cluster_centers_[1]
