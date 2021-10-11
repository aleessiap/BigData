from pyspark.ml.feature import HashingTF, Tokenizer, IDF, StopWordsRemover, RegexTokenizer, Bucketizer, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType, NumericType, IntegerType
from pyspark.ml import Pipeline
from sparknlp.base import *
from sparknlp.annotator import *
import sparknlp
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
import pyspark.sql.functions as f
from preprocessing import replace_contraction, stop_words, getOriginalSentiment, map_overall_to_sentiment
import pyspark
from pyspark.sql import SparkSession

from nltk.corpus import wordnet



def get_wordnet_pos(data):
    wnl = WordNetLemmatizer()
    lemmas = []
    for word, tag in pos_tag(word_tokenize(data)):

        wntag = tag[0].lower()

        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None

        if not wntag:

            lemma = word
        else:

            lemma = wnl.lemmatize(word, wntag)

        lemmas.append(lemma)
        #print(lemma)

    return lemmas

def repl_contr(list):
    return ' '.join([replace_contraction(word) for word in list])

def concat_tokens(list):
    return ' '.join([word for word in list])

def tokenize(data):
    tknz = TweetTokenizer()

    return tknz.tokenize(data)

def remove_digits(data):
    return re.sub(r'[' + string.digits + ']+', '', data)

def remove_short(data):
    return re.sub(r'\b\w{1,3}\b', '', data)

def remove_puntaction(data):
    return re.sub(r'[' + string.punctuation + ']', ' ', data)

def remove_whiteSpace(data):
    return re.sub(r'[' + string.whitespace + ']+', ' ', data)

def remove_stop(data):
    return [word for word in data if word not in stop_words]

def lemmatize(data):
    return get_wordnet_pos(data)

def get_sentiment(data):

    return 1 if data > 3.0 else 0 if data == 3.0 else 2

def trainW2v(wordsData, trainPerc, numIstances):
    testPerc = 1 - trainPerc
    word2vec = Word2Vec(vectorSize=300, minCount=1, inputCol='tokens_without_stop', outputCol='word2vec_features')
    wordsData = word2vec.fit(wordsData).transform(wordsData)
    wordsData = wordsData.drop('reviewText', 'tokens', 'reviews_notcontracted', 'reviews_without_digits', 'reviews_without_puntaction', 'reviews_without_white', 'lemmas','tokens_without_stop')

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
    f = open('C:\\Users\\Alessia\\PyCharmProjects\\BigData\\models\\'+name, "a")
    f.write('f1: ' + str(evaluator_f1.evaluate(lr_predictions_w2v)))
    f.write('\n')
    f.write('acc: ' + str(evaluator_acc.evaluate(lr_predictions_w2v)))
    f.write('\n')
    f.write('recall: ' + str(evaluator_recall.evaluate(lr_predictions_w2v)))
    f.write('\n')
    f.write('precision: ' + str(evaluator_precision.evaluate(lr_predictions_w2v)))
    f.close()


def Berttrain(wordsData, trainPerc, numInstances):
    # include our preprocessing i don't now if it make sense

    sent_udf = f.udf(concat_tokens, StringType())
    wordsData = wordsData.withColumn('phrase_to_analize', sent_udf('tokens_without_stop'))

    wordsData = wordsData.drop('reviewText', 'tokens_without_stop', 'tokens', 'reviews_notcontracted',
                               'reviews_without_digits', 'reviews_without_puntaction', 'reviews_without_white',
                               'lemmas', 'tokens_without_stop')
    pp.pprint(wordsData.head(10))

    documentAssembler = DocumentAssembler() \
        .setInputCol("phrase_to_analize") \
        .setOutputCol("document")
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    word_embeddings = BertEmbeddings.pretrained('bert_wiki_books', 'en') \
        .setInputCols(["document", "token"]) \
        .setOutputCol("embeddings")

    embeddingsSentence = SentenceEmbeddings() \
        .setInputCols(["document", "embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")

    classifierdl = ClassifierDLApproach() \
        .setInputCols(["sentence_embeddings"]) \
        .setOutputCol("prediction") \
        .setLabelColumn("label") \
        .setEnableOutputLogs(True)

    bert_pipeline = Pipeline().setStages(
        [
            documentAssembler,
            tokenizer,
            word_embeddings,
            embeddingsSentence,
            classifierdl
        ]
    )

    df_bert = bert_pipeline.fit(wordsData).transform(wordsData)
    
    pp.pprint(df_bert.head())
    pp.pprint(df_bert.dtypes)


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


if __name__ == '__main__':
    #spark = SparkSession.builder.getOrCreate()
    #sc = spark.sparkContext
    sparknlp = sparknlp.start()

    pp = pprint.PrettyPrinter(indent=4)

    dataset = sparknlp.read.json('C:\\Users\\Alessia\\PyCharmProjects\\BigData\\dataset\\test.json')
    # add an index column
    df = dataset.withColumn('index', f.monotonically_increasing_id())

    # sort ascending and take first 100 rows for df1

    for n in [40000]:
        for tr in [0.9]:
           wordsData = preprocessing(df, n)
           # train(wordsData, tr, n)
           Berttrain(wordsData, tr, n)
           # pp.pprint(wordsData.head(3))



    # regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words_token", pattern="\\W")
    # wordsData = regexTokenizer.transform(df1)

    # df1 = df1.withColumn('reviewText', ' '.join([replace_contraction(word=word) for word in words ]))
