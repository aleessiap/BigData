import time
import string

import sparknlp

from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.ml import Pipeline
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag, word_tokenize
from pyspark.sql.types import StringType, ArrayType, IntegerType
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import pprint
import sys
import pyspark.sql.functions as f


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

stop_words = ['to', 'and', 'or', 'be', 'have', 'in', 'of', 'at', 'a', 'an', 'the', 'i', 'we', 'you', 'they', 'he',
              'she', 'it',
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
        # print(lemma)

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


def trainBert(wordsData, trainPerc, n, file):
    testPerc = 1 - trainPerc

    sent_udf = f.udf(concat_tokens, StringType())
    wordsData = wordsData.withColumn('phrase_to_analyze', sent_udf('tokens_without_stop'))

    wordsData = wordsData.drop('reviewText', 'tokens_without_stop', 'tokens', 'reviews_notcontracted',
                               'reviews_without_digits', 'reviews_without_puntaction', 'reviews_without_white',
                               'lemmas', 'tokens_without_stop')
    pp.pprint(wordsData.head(1))

    documentAssembler = DocumentAssembler() \
        .setInputCol("phrase_to_analyze") \
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("tokens")

    word_embeddings = BertEmbeddings.pretrained('bert_wiki_books', 'en') \
        .setInputCols(["document", "tokens"]) \
        .setOutputCol("embeddings")

    embeddingsSentence = SentenceEmbeddings() \
        .setInputCols(["document", "embeddings"]) \
        .setOutputCol("sentence_embeddings") \
        .setPoolingStrategy("AVERAGE")

    embeddingsFinisher = EmbeddingsFinisher()\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCols("finishedEmbeddings")\
        .setOutputAsVector(True)

    classsifierdl = ClassifierDLApproach() \
        .setInputCols(["sentence_embeddings"]) \
        .setOutputCol("class") \
        .setLabelColumn("label") \
        .setMaxEpochs(50) \
        .setEnableOutputLogs(True)

    bert_pipeline = Pipeline().setStages(
        [
            documentAssembler,
            tokenizer,
            word_embeddings,
            embeddingsSentence,
            embeddingsFinisher,
            classsifierdl
        ]
    )

    start = time.time_ns()
    (trainingData, testData) = wordsData.randomSplit([trainPerc, testPerc], seed=100)
    end = time.time_ns()
    print("Splitting train - test: " + str((end - start) / 1000000000))
    train_count = str(trainingData.count())
    test_count = str(testData.count())
    print("Training Dataset Count: " + train_count)
    print("Test Dataset Count: " + test_count)
    file.write("Training Dataset Count: " + train_count + '\n')
    file.write("Test Dataset Count: " + test_count + '\n')

    if n != -1:
        bert_pipeline = bert_pipeline.repartition(n)

    model = bert_pipeline.fit(trainingData)
    # pp.pprint(model.transform(trainingData))
    predictions = model.transform(testData)
    pp.pprint(predictions)
    df_bert = predictions.select("label", "phrase_to_analyze", "class.result") \
        .toPandas()

    df_bert['result'] = df_bert.result.str[0].astype('int32')
    file.write('acc: ' + str(accuracy_score(df_bert.label, df_bert.result)))
    file.write('\n')
    file.write('\nMICRO\n')
    file.write('f1: ' + str(f1_score(df_bert.label, df_bert.result, average='micro')))
    file.write('\n')
    file.write('recall: ' + str(recall_score(df_bert.label, df_bert.result, average='micro')))
    file.write('\n')
    file.write('precision: ' + str(precision_score(df_bert.label, df_bert.result, average='micro')))
    file.write('\n')
    file.write('\nMACRO\n')
    file.write('f1: ' + str(f1_score(df_bert.label, df_bert.result, average='macro',  zero_division=0)))
    file.write('\n')
    file.write('recall: ' + str(recall_score(df_bert.label, df_bert.result, average='macro',  zero_division=0)))
    file.write('\n')
    file.write('precision: ' + str(precision_score(df_bert.label, df_bert.result, average='macro',  zero_division=0)))
    file.write('\n')
    file.write(classification_report(df_bert.label, df_bert.result,  zero_division=0))
    file.write('\n')


if __name__ == '__main__':

    sparknlp = sparknlp.start(gpu=True)

    pp = pprint.PrettyPrinter(indent=4)
    time_start = time.time_ns()
    # dataset = sparknlp.read.json('/home/ubuntu/BigData/dataset/Kindle.json')

    dataset = sparknlp.read.json('./dataset/Kindle.json')

    df = dataset.withColumn('index', f.monotonically_increasing_id())

    numRepartitions = -1
    numReviews = 10000
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

    name = 'Bert-Kindle-' + str(trainPerc) + '-' + str(numReviews) + '-3classi.txt'
    # file = open('/home/ubuntu/BigData/experiments/' + name, "a")
    file = open('./experiments/' + name, "a")

    start = time.time_ns()
    wordsData = preprocessing(df, numReviews, numRepartitions)
    end = time.time_ns()
    print("Preprocessing time in s: " + str((end - start) / 1000000000))
    trainBert(wordsData, trainPerc, numRepartitions, file)
    time_end = time.time_ns()
    time = time_end - time_start
    print("Total time in s: ")
    time_final = time / 1000000000
    print(time_final)
    file.write("Total time in s:")
    file.write(str(time_final))
    file.close()
