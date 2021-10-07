import string

from pyspark.mllib.feature import Word2Vec, Word2VecModel
from nltk.corpus import brown, webtext, gutenberg, abc, movie_reviews, product_reviews_1, product_reviews_2, sentence_polarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import re
from preprocessing import replace_contraction

stop_words = ['to', 'and', 'or', 'be', 'in', 'of', 'at', 'a', 'an', 'the', 'i', 'we', 'you', 'they', 'he', 'she', 'it',
              'that', 'this', 'those', 'these', 'by', 'my', 'your', 'our', 'his', 'her', 'its', 'their', 'yourself',
              'yourselves', 'yet', 'us', 'upon', 'until']


def getWord2Vec(sc):
    listProd1 = sc.parallelize(list(product_reviews_1.sents()))
    listProd2 = sc.parallelize(list(product_reviews_2.sents()))
    listSent = sc.parallelize(list(sentence_polarity.sents()))

    listMovie = sc.parallelize(list(movie_reviews.sents()))
    listAbc = sc.parallelize(list(abc.sents()))
    listBrown = sc.parallelize(list(brown.sents()))
    listWeb = sc.parallelize(list(webtext.sents()))
    listGut = sc.parallelize(list(gutenberg.sents()))
    listProd1 = listProd1.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    listProd2 = listProd2.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    listSent = listSent.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))

    listMovie = listMovie.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    listAbc = listAbc.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    listGut = listGut.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    listBrown = listBrown.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    listWeb = listWeb.map(lambda x: ' '.join(x)).map(lambda x: preprocess(x))
    data = listBrown.union(listWeb).union(listGut).union(listAbc).union(listMovie).union(listProd1).union(listProd2)\
        .union(listSent)
    return data


def preprocess(data):
    # remove urls
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)
    # print(text)
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

    # remove punctuation
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

    # print(text)

    return text


