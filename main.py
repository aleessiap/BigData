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
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.classification import LogisticRegressionModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
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


def createModel(sc, trainSet):
    word2vec = Word2Vec().setVectorSize(300).setMinCount(100)
    model = word2vec.fit(trainSet)
    model.save(sc, 'C:\\Users\\Alessia\\PycharmProjects\\BigData\\models\\ModelWith90Kindle.model')
    print(list(model.findSynonyms("good", 10)))
    print(list(model.findSynonyms("bad", 10)))
    print(list(model.findSynonyms("perfect", 10)))
    print(list(model.findSynonyms("horrible", 10)))
    print(list(model.findSynonyms("dog", 10)))
    print(list(model.findSynonyms("book", 10)))

    # sentiment = getOriginalSentiment(sc, dataset)
    return model


def Kmeans(model, model_name, vectors, num, steps):
    initsteps = 100
    maxIteration = 1000

    name = str(model_name) + '_' + str(num) + '-' + str(steps) + '.txt'
    f = open(name, "a")

    clustering = KMeans()
    clusterTrained = clustering.train(rdd=vectors, k=num, maxIterations=maxIteration, initializationSteps=steps)
    positive_cluster_center = clusterTrained.clusterCenters[0]
    negative_cluster_center = clusterTrained.clusterCenters[1]

    centroids = clusterTrained.centers
    for i in range(0, num):
        print(list(model.findSynonyms(centroids[i], 10)))
        f.write(' \n')
        f.write(str(list(model.findSynonyms(centroids[i], 10))))
        f.write('\n')
    f.close()


def clusteringPhase(sc, modelName):
    model = Word2VecModel.load(sc, "C:\\Users\\Alessia\\PycharmProjects\\BigData\\models\\" + modelName)

    # vectors = PCA(vectors)
    for k in [70, 100, 120]:
        for s in [200, 220]:
            Kmeans(model, modelName, vectors, k, s)

    return


def PCA(vectors):
    mat = RowMatrix(vectors)
    pc = mat.computePrincipalComponents(2)
    # pc = sc.parallelize(pc)
    print(pc)
    print(type(pc))
    print(pc.apply(1, 1))
    print(type(pc.apply(1, 1)))
    return pc


def classification(trainSet, testSet):
    lr = LogisticRegressionModel(numClasses=3)
    lr_model = lr.fit(trainSet)
    lr_predictions = lr_model.transform(testSet)
    print(lr_predictions)


if __name__ == '__main__':
    spark, sc = init_spark()
    df = spark.sql('select * from wine1')

    trainSet, testSet = read_dataset(sc, dataset, 90, 20)
    model = Word2VecModel.load(sc, "C:\\Users\\Alessia\\PycharmProjects\\BigData\\models\\ModelWith90Kindle.model")
    # print(model.getVectors().values())
    dicVectors = dict(model.getVectors())
    values = []
    for key in dicVectors:
        values.append(list(dicVectors[key]))
    vectors = sc.parallelize(values, numSlices=100)
    print(type(vectors))
    # model = createModel(sc, trainSet)
    # clusteringPhase(sc, "ModelWith90Kindle.model")
    classification(vectors, testSet)
    spark.stop()
