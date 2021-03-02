import string
import json
import nltk
import sys
import re
import os

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def init_spark():

    spark = SparkSession.builder.appName("HelloWorld").getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():

    # spark, sc = init_spark()
    # nums = sc.parallelize([1, 2, 3, 4])
    # print(nums.map(lambda x: x * x).collect())

    spark, sc = init_spark()
    dataset = sc.textFile(name='dataset/Appliances_5.json')
    # dataset = sc.textFile(name='C:\\Users\\Alessia\\PycharmProjects\\BigData2\\dataset\\Appliances_5.json')
    print(dataset.map(lambda x: json.loads(x)).map(lambda x: (clean_text(x['reviewText']), x['overall'])).collect())


def clean_text(text):

    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

    text = re.sub(r'[' + string.digits + ']+', '', text)
    text = re.sub(r'[' + string.punctuation + ']', '', text)
    text = re.sub(r'[' + string.whitespace + ']+', ' ', text)
    text = text.lower()

    text = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    from nltk.corpus import stopwords

    text = [word for word in text if word not in stopwords.words('english')]
    text = [lemmatizer.lemmatize(word, pos='v') for word in text]
    text = [lemmatizer.lemmatize(word, pos='n') for word in text]

    return text


if __name__ == '__main__':

    main()

