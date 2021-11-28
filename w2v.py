import time
import string
import json
import nltk
import sys
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize
from pyspark.sql.types import StringType, ArrayType, NumericType, IntegerType
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import NGram
import pprint
import sys
import pyspark.sql.functions as f
from pyspark.sql.session import SparkSession

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
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
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
    "mustnt": "must not",
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
    "shes": "she is",
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
    "theyll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "wasnt": "was not",
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
    "whens": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "wheres": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "whos": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "willve": "will have",
    "won't": "will not",
    "wont": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldve": "would have",
    "wouldn't": "would not",
    "wouldnt": "would not",
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

stop_words = ['to', 'and', 'or', 'be', 'have', 'in', 'of', 'at', 'a', 'an', 'the', 'i', 'we', 'you', 'they', 'he', 'she', 'it',
              'that', 'this', 'those', 'these', 'by', 'my', 'your', 'our', 'his', 'her', 'its', 'their', 'me']

def replace_contraction(word):
    ret = word

    if word in CONTRACTION_MAP.keys():
        ret = CONTRACTION_MAP[word]

    return ret

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


def preprocessing(df, numInstances, numRepartitions):

    if numRepartitions == -1:
        df1 = df.sort('index').limit(numInstances)
    else:
        df1 = df.sort('index').limit(numInstances).repartition(numRepartitions)

    df1 = df1.dropna(subset=('reviewText', 'overall')).select('*')
    get_sentiment_udf = f.udf(get_sentiment, IntegerType())
    df1 = df1.withColumn('label', get_sentiment_udf('overall'))

    df1 = df1.select('label', (df1['reviewText']).alias('reviewText'))

    df1 = df1.withColumn('reviewText', f.lower(df1.reviewText))
    tok_udf = f.udf(tokenize, ArrayType(StringType()))

    wordsData = df1.withColumn('tokens', tok_udf('reviewText'))

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

    remove_stop_udf = f.udf(remove_stop, ArrayType(StringType()))
    wordsData = wordsData.withColumn('tokens_without_stop', remove_stop_udf('lemmas'))

    wordsData = wordsData.drop('reviewText', 'tokens', 'reviews_notcontracted', 'reviews_without_digits',
                               'reviews_without_puntaction', 'reviews_without_white', 'lemmas')

    return wordsData


def trainW2v(wordsData, trainPerc, n, file):
    testPerc = 1 - trainPerc

    word2vec = Word2Vec(vectorSize=300, minCount=1, inputCol='tokens_without_stop', outputCol='word2vec_features')

    if n == -1:
        start = time.time_ns()
        wordsData = word2vec.fit(wordsData).transform(wordsData)
    else:
        wordsData = wordsData.repartition(n)
        start = time.time_ns()
        wordsData = word2vec.fit(wordsData).transform(wordsData)

    end = time.time_ns()
    print("Embedding time in s: " + str((end - start)/1000000000))
    wordsData = wordsData.drop('tokens_without_stop')

    start = time.time_ns()
    (trainingData_w2v, testData_w2v) = wordsData.randomSplit([trainPerc, testPerc], seed=100)
    end = time.time_ns()
    print("Splitting train - test: " + str((end - start) / 1000000000))
    train_count = str(trainingData_w2v.count())
    test_count = str(testData_w2v.count())
    print("Training Dataset Count: " + train_count)
    print("Test Dataset Count: " + test_count)
    file.write("Training Dataset Count: " + train_count + '\n')
    file.write("Test Dataset Count: " + test_count + '\n')

    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='f1')
    evaluator_acc = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='accuracy')
    evaluator_recall = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='weightedRecall')
    evaluator_precision = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='weightedPrecision')

    lr_w2v = LogisticRegression(featuresCol='word2vec_features').setFamily('multinomial')

    start = time.time_ns()
    lr_Model_w2v = lr_w2v.fit(trainingData_w2v)
    lr_predictions_w2v = lr_Model_w2v.transform(testData_w2v)
    end = time.time_ns()
    print("Classification time in s: " + str((end - start)/1000000000))

    start = time.time_ns()
    f1 = str(evaluator_f1.evaluate(lr_predictions_w2v))
    acc = str(evaluator_acc.evaluate(lr_predictions_w2v))
    recall = str(evaluator_recall.evaluate(lr_predictions_w2v))
    precision = str(evaluator_precision.evaluate(lr_predictions_w2v))

    end = time.time_ns()
    print("Evaluating time in s: " + str((end - start)/1000000000))

    print('f1: ' + f1)
    file.write('f1: ' + f1 + '\n')
    print('acc: ' + acc)
    file.write('acc: ' + acc + '\n')
    print('recall: ' + recall)
    file.write('recall: ' + recall + '\n')
    print('precision: ' + precision)
    file.write('precision: ' + precision + '\n')


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext

    pp = pprint.PrettyPrinter(indent=4)
    time_start = time.time_ns()
    #dataset = spark.read.json('/home/ubuntu/BigData/dataset/Kindle.json')

    dataset = spark.read.json('./dataset/Kindle.json')

    df = dataset.withColumn('index', f.monotonically_increasing_id())

    numRepartitions = -1
    numReviews = dataset.count()
    trainPerc = 0.9
    n = len(sys.argv)

    for i in range(1, n):
        if i == 1:
            numRepartitions = int(sys.argv[i])

        elif i == 2:
             numReviews = int(sys.argv[i])

        elif i == 3:
            trainPerc = float(sys.argv[i])

    print('train ' + str(trainPerc))
    print('repartitions: ' + str(numRepartitions))
    print('reviews: ' + str(numReviews))

    name = 'LogisticRegression-Kindle-' + str(trainPerc) + '-' + str(numReviews) + '-3classi.txt'
    #file = open('/home/ubuntu/BigData/experiments/' + name, "a")
    file = open('./experiments/' + name, "a")

    start = time.time_ns()
    wordsData = preprocessing(df, numReviews, numRepartitions)
    end = time.time_ns()
    print("Preprocessing time in s: " + str((end - start)/1000000000))
    trainW2v(wordsData, trainPerc, numRepartitions, file)
    time_end = time.time_ns()
    time = time_end - time_start
    print("Total time in s: ")
    time_final = time/1000000000
    print(time_final)
    file.write("Total time in s:")
    file.write(str(time_final))
    file.close()
