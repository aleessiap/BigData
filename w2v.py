from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover, RegexTokenizer, Bucketizer, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType, NumericType, IntegerType
from pyspark.ml import Pipeline

import string
import json
import nltk
import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize

from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import NGram
import pprint
from trying_thing import *
import pyspark.sql.functions as f
from preprocessing import replace_contraction, stop_words, getOriginalSentiment, map_overall_to_sentiment


def preprocessing(df, numInstances):
    df1 = df.sort('index').limit(numInstances)

    df1 = df1.dropna(subset=('reviewText', 'overall')).select('*')
    get_sentiment_udf = f.udf(get_sentiment, IntegerType())
    df1 = df1.withColumn('label', get_sentiment_udf('overall'))

    df1 = df1.select('label',  (df1['reviewText']).alias('reviewText'))
    # df1.printSchema()
    # pp.pprint(df1.head(5))
    df1 = df1.withColumn('reviewText', lower(df1.reviewText))
    tok_udf = f.udf(tokenize, ArrayType(StringType()))
    # regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words_token", pattern="\\W")
    # wordsData = regexTokenizer.transform(df1)
    wordsData = df1.withColumn('tokens', tok_udf('reviewText'))
    # remove_stop_udf = f.udf(remove_stop, ArrayType(StringType()))
    # wordsData = wordsData.withColumn('tokens_without_stop', remove_stop_udf('tokens'))
    rudf = f.udf(repl_contr, StringType())
    wordsData = wordsData.withColumn('reviews_notcontracted', rudf('tokens'))

    remove_digits_udf = f.udf(remove_digits, StringType())
    wordsData = wordsData.withColumn('reviews_without_digits', remove_digits_udf('reviews_notcontracted'))

    remove_punt_udf = f.udf(remove_puntaction, StringType())
    wordsData = wordsData.withColumn('reviews_without_puntaction', remove_punt_udf('reviews_without_digits'))

    remove_white_udf = f.udf(remove_whiteSpace, StringType())
    wordsData = wordsData.withColumn('reviews_without_white', remove_white_udf('reviews_without_puntaction'))

    lemmatize_udf = f.udf(lemmatize, ArrayType(StringType()))
    wordsData = wordsData.withColumn('lemmas', lemmatize_udf('reviews_without_white'))
    # lemmatizer = WordNetLemmatizer()
    remove_stop_udf = f.udf(remove_stop, ArrayType(StringType()))
    wordsData = wordsData.withColumn('tokens_without_stop', remove_stop_udf('lemmas'))

    return wordsData


def trainW2v(wordsData, trainPerc, numIstances):
    testPerc = 1 - trainPerc
    word2vec = Word2Vec(vectorSize=300, minCount=1, inputCol='tokens_without_stop', outputCol='word2vec_features')
    wordsData = word2vec.fit(wordsData).transform(wordsData)
    wordsData = wordsData.drop('reviewText', 'tokens', 'reviews_notcontracted', 'reviews_without_digits',
                               'reviews_without_puntaction', 'reviews_without_white', 'lemmas','tokens_without_stop')

    # dataset_w2v = dataset_w2v.withColumnRenamed("overall", "label")
    (trainingData_w2v, testData_w2v) = wordsData.randomSplit([trainPerc, testPerc], seed=100)
    print("Training Dataset Count: " + str(trainingData_w2v.count()))
    print("Test Dataset Count: " + str(testData_w2v.count()))

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='f1')
    evaluator_acc = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='accuracy')
    evaluator_recall = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='weightedRecall')
    evaluator_precision = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='weightedPrecision')

    lr_w2v = LogisticRegression(featuresCol='word2vec_features').setFamily('multinomial')
    # lr_mmlib = LogisticRegressionModel(features)
    lr_Model_w2v = lr_w2v.fit(trainingData_w2v)
    lr_predictions_w2v = lr_Model_w2v.transform(testData_w2v)
    name = 'LogisticRegression-Kindle-'+str(trainPerc)+'-'+str(testPerc)+'-'+str(numIstances)+'3classi.txt'
    f = open('.\models\\'+name, "a")
    f.write('f1: ' + str(evaluator_f1.evaluate(lr_predictions_w2v)))
    f.write('\n')
    f.write('acc: ' + str(evaluator_acc.evaluate(lr_predictions_w2v)))
    f.write('\n')
    f.write('recall: ' + str(evaluator_recall.evaluate(lr_predictions_w2v)))
    f.write('\n')
    f.write('precision: ' + str(evaluator_precision.evaluate(lr_predictions_w2v)))
    f.close()


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    pp = pprint.PrettyPrinter(indent=4)

    dataset = spark.read.json('.\dataset\\Kindle.json')
    df = dataset.withColumn('index', f.monotonically_increasing_id())

    wordsData = preprocessing(df, 65000)
    trainW2v(wordsData, 0.9, 65000)
