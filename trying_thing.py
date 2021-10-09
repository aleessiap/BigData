from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover, RegexTokenizer, Bucketizer, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
import string
import json
import nltk
import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import NGram
import pprint
import pyspark.sql.functions as f
from preprocessing import replace_contraction, stop_words
import pyspark
from pyspark.sql import SparkSession


def print_eval_w2v(metrics, name_of_feature):
    print(name_of_feature)


# print('f1: ' + str(evaluator_f1.evaluate(metrics)))
# print('acc: ' + str(evaluator_acc.evaluate(metrics)))
# print('recall: ' + str(evaluator_recall.evaluate(metrics)))
# print('precision: ' + str(evaluator_precision.evaluate(metrics)))

def train(dataset):
    # df_clean_null = dataset.dropna(subset=('reviewText', 'overall')).select('*')
    df_lower = df_clean_null.select('overall', (df_clean_null['reviewText']).alias('reviewText'))
    regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words_token", pattern="\\W")

    # stopwordsRemover = StopWordsRemover(inputCol="words_token", outputCol="no_stop_words")

    wordsData = regexTokenizer.transform(df_lower)
    # noStopWords = stopwordsRemover.transform(wordsData)
    # print(noStopWords.head(20))

    word2vec = Word2Vec(vectorSize=300, minCount=1, inputCol='no_stop_words', outputCol='word2vec_features', )
    features = word2vec.fit(wordsData).transform(wordsData)

    dataset_w2v = features.drop('reviewText', 'words_token', 'no_stop_words')
    dataset_w2v = dataset_w2v.withColumnRenamed("overall", "label")
    (trainingData_w2v, testData_w2v) = dataset_w2v.randomSplit([0.8, 0.2], seed=100)
    print("Training Dataset Count: " + str(trainingData_w2v.count()))
    print("Test Dataset Count: " + str(testData_w2v.count()))

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='f1')
    evaluator_acc = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='accuracy')
    evaluator_recall = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='weightedRecall')
    evaluator_precision = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='weightedPrecision')

    lr_w2v = LogisticRegression(featuresCol='word2vec_features')
    # lr_mmlib = LogisticRegressionModel(features)
    lr_Model_w2v = lr_w2v.fit(trainingData_w2v)
    lr_predictions_w2v = lr_Model_w2v.transform(testData_w2v)

    print('f1: ' + str(evaluator_f1.evaluate(lr_predictions_w2v)))
    print('acc: ' + str(evaluator_acc.evaluate(lr_predictions_w2v)))
    print('recall: ' + str(evaluator_recall.evaluate(lr_predictions_w2v)))
    print('precision: ' + str(evaluator_precision.evaluate(lr_predictions_w2v)))


def repl_contr(list):

    return ' '.join([replace_contraction(word) for word in list])


def preprocess(data):
    print(type(data))
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


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()

    dataset = spark.read.json('C:\\Users\\frcon\\Desktop\\BigData\\dataset\\test.json')
    # add an index column
    #df = dataset.withColumn('index', f.monotonically_increasing_id())

    # sort ascending and take first 100 rows for df1
    #df1 = df.sort('index').limit(50000)
    df1 = dataset.dropna(subset=('reviewText', 'overall')).select('*')
    df1 = df1.withColumn('reviewText', lower(df1.reviewText))
    regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words_token", pattern="\\W")
    wordsData = regexTokenizer.transform(df1)


    print(wordsData.select(f.collect_list('words_token')).first())

    print(wordsData)
    rudf = f.udf(repl_contr, StringType())
    print(wordsData.withColumn('reviewText', rudf('words_token')))

    regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words_token", pattern="\\W")
    wordsData = regexTokenizer.transform(df1)

    #df1 = df1.withColumn('reviewText', ' '.join([replace_contraction(word=word) for word in words ]))





    #print(df1.head(1))
