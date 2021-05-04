from __future__ import print_function
from datetime import datetime
from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from pprint import pprint
import codecs
import collections
import json
import nltk
import nltk.util
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.layers import Input
from keras.layers import Flatten, LSTM
from keras.models import Model
from numpy import array
from numpy import asarray
from numpy import zeros
import pickle

df = pd.read_csv('dataframe.csv')

with open('tokenizer.pickle', 'rb') as f_obj:
    word_tokenizer = pickle.load(f_obj)

with open("person_umbrellas_subterms.pickle", "rb") as fobj:
    person_umbrellas_subterms = pickle.load(fobj)

bar_plot = pd.DataFrame()
bar_plot['cat'] = df.columns[2:]
bar_plot['count'] = df.iloc[:,2:].sum().values
bar_plot.sort_values(['count'], inplace=True, ascending=False)
bar_plot.reset_index(inplace=True, drop=True)


main_categories = pd.DataFrame()
main_categories = bar_plot[bar_plot['count']>0]
categories = main_categories['cat'].values
categories = np.append(categories,'Others')
not_category = []
df['Others'] = 0

for i in df.columns[2:]:
    if i not in categories:
        df['Others'][df[i] == 1] = 1
        not_category.append(i)

df.drop(not_category, axis=1, inplace=True)

most_common_cat = pd.DataFrame()
most_common_cat['cat'] = df.columns[2:]
most_common_cat['count'] = df.iloc[:,2:].sum().values
most_common_cat.sort_values(['count'], inplace=True, ascending=False)
most_common_cat.reset_index(inplace=True, drop=True)

threshold = 200


columns = ['OneVsAll', 'BinaryRelevance', 'ClassifierChain', 'MultipleOutput','FFN', 'CNN', 'LSTM']
results = pd.DataFrame(columns = columns)

seeds = [1, 43, 678, 90, 135]

t = results.copy()

# Preprocessing

def decontract(sentence):
    # specific
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

def cleanPunc(sentence):
    cleaned = [re.sub(r'[?|!|\'|"|#]',r'',word) for word in sentence]
    cleaned = [re.sub(r'[.|,|)|(|\|/]',r' ',word) for word in cleaned]
    cleaned = [word.strip() for word in cleaned]
    cleaned = [word.replace("\n"," ") for word in cleaned]
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

stop_words = stopwords.words('english')

stop_words.extend(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'also', 'however', 'could', 'like', 'therefore', 'may', 'quite', '\'s', 'able', 'must', 'often', 'since', \
            'whether', 'unless', 'upon', 'even', 'thus', 'in', 'on', 'when', 'under', 'using', 'without', \
            'won', "won't", 'wouldn', "wouldn't", 'also', 'however', 'could', 'like', 'therefore', 'may', 'quite', '\'s', 'able', 'must', \
            'often', 'since', 'whether', 'unless', 'upon', 'even', 'thus', 'in', 'on', 'when', 'under', \
            'using', 'without', "'s", 'abl', 'abov', 'ain', 'ani', 'aren', "aren't", 'becaus', 'befor', \
            'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'doe', 'doesn', "doesn't", 'don', "don't", \
            'dure', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'howev', 'isn', "isn't", 'just', \
            'like', 'll', 'm', 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'o', 'onc', \
            'onli', 'ourselv', 'quit', 's', 'shan', "shan't", "should'v", 'shouldn', "shouldn't", 'sinc', 't', \
            "that'll", 'themselv', 'therefor', 'unless', 'use', 've', 'veri', 'wasn', "wasn't", 'weren', "weren't", \
            'whi', 'won', "won't", 'wouldn', "wouldn't", 'y', "you'd", "you'll", "you'r", "you'v", 'yourselv', 'self', 'of'])

def removeStopWords(sentence):
    stopwords_removed = [word for word in sentence if word not in stop_words]
    return stopwords_removed

stemmer = SnowballStemmer("english")

def stemming(sentence):
    stems = [stemmer.stem(word) for word in sentence]
    return stems

def tags(sentence):
    remove_spans = re.sub(r'</?span>', ' ', sentence)

    remove_non_alphanumeric = re.sub(r'\W', ' ', remove_spans)
    remove_numbers = re.sub(r'\d+', ' ', remove_non_alphanumeric)
    try:
        tokens = word_tokenizer.tokenize(remove_numbers.split())
    except:
        import pdb; pdb.set_trace()
    remove_underscores = [token.replace("_", " ") for token in tokens]

    pos = nltk.pos_tag(remove_underscores)
    remove_verbs = [pos_tuple for pos_tuple in pos
                    if pos_tuple[1] != 'VBZ'
                    and pos_tuple[1] != 'RB'
                    and pos_tuple[1] != 'VBN'
                    and pos_tuple[1] != 'VBD'
                    and pos_tuple[1] != 'VBG']
    ne = nltk.ne_chunk(remove_verbs)

    removed_ne = [item[0] for item in ne if not isinstance(item, nltk.tree.Tree)]

    return removed_ne


def preprocess(sentence):
    sentence = tags(sentence)
    sentence = [word.lower() for word in sentence]
    sentence = cleanPunc(sentence)
    sentence = removeStopWords(sentence)
    sentence = stemming(sentence)
    return sentence

df['description'] = df['description'].apply(preprocess)

# MODELS

# FFN

tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(df['description'])
sequences = tokenizer.texts_to_sequences(df['description'])
x = pad_sequences(sequences, maxlen=200)

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    df[df.columns[2:]],
                                                    test_size=0.3,
                                                    random_state=seeds[4])

most_common_cat['class_weight'] = len(most_common_cat) / most_common_cat['count']
class_weight = {}
for index, label in enumerate(categories):
    class_weight[index] = most_common_cat[most_common_cat['cat'] == categories]['class_weight'].values[0]

most_common_cat.head()

num_classes = y_train.shape[1]
max_words = len(tokenizer.word_index) + 1
maxlen = 200

model = Sequential()
model.add(Embedding(max_words, 20, input_length=maxlen))
#model.add(Dropout(0.2))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])


callbacks = [
    ReduceLROnPlateau(),
    #EarlyStopping(patience=10),
    ModelCheckpoint(filepath='model-simple.h5', save_best_only=True)
]


history = model.fit(X_train, y_train,
                    class_weight=class_weight,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.3,
                    callbacks=callbacks)


dnn_model = model
metrics = dnn_model.evaluate(X_test, y_test)
print("FFN")
print("{}: {}".format(dnn_model.metrics_names[1], metrics[1]))

results.loc[4,'FFN'] = metrics[1]


def predict(text, model):
    text = preprocess(text)
    sequences = tokenizer.texts_to_sequences([text])
    input_data = pad_sequences(sequences, maxlen=200)
    return model.predict(input_data)[0]

def print_pred(pred):
    prof_categories = []
    for value, category in zip(pred, categories):
        prof_categories.append(re.sub(r',.+', '', category))
    return prof_categories

pred1 = predict(pd.read_csv('dataframe.csv')['description'][0], dnn_model)
print(print_pred(pred1))

# CNN


filter_length = 300

model = Sequential()
model.add(Embedding(max_words, 20, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]

history = model.fit(X_train, y_train,
                    class_weight=class_weight,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.3,
                    callbacks=callbacks)

cnn_model = model
metrics = cnn_model.evaluate(X_test, y_test)
print("CNN")
print("{}: {}".format(model.metrics_names[1], metrics[1]))

results.loc[4,'CNN'] = metrics[1]
print(results)

# GLOVE - LSTM

embeddings_dictionary = dict()

glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((max_words, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(max_words, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(94)(embedding_layer)
dense_layer_1 = Dense(240, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

callbacks = [
    ReduceLROnPlateau(),
    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
history = model.fit(X_train, y_train.values,
                    class_weight=class_weight,
                    batch_size=32,
                    epochs=30,
                    validation_split=0.3,
                    callbacks=callbacks)

metrics = model.evaluate(X_test, y_test)
print("LSTM")
print("{}: {}".format(model.metrics_names[1], metrics[1]))

results.loc[4,'LSTM'] = metrics[1]

print(results)
print()

