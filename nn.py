import nltk 
from nltk.corpus import stopwords
from keras import models, layers
import heapq
# import glove

import argparse
import zipfile
import sklearn.metrics
import pandas as pd
import numpy as np
import json
import re
import emoji
import string


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}


def train_and_predict(train_word_data, train_labels,
                      test_word_data, test_labels, vocab_len):

    # doesn't train anything; just predicts 1 for all of dev set
    # dev_predictions = dev_data.copy()
    # dev_predictions[emotions] = 1

    model, kwargs = create_model(vocab_len)

    print("train_word[0]", len(train_word_data[0]))
    print("train_word", len(train_word_data))

    kwargs.update(x=train_word_data, y=train_labels,
                  epochs=10, validation_data=(test_word_data, test_labels))

    model.fit(**kwargs)

    return model.predict(test_word_data, test_labels)

def create_model(vocab_len):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_len, 200, mask_zero=True))
    model.add(layers.Bidirectional(layers.GRU(64)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(20, activation ='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return [model, {}]

def preprocess_data(train, test):
    # print(train['Tweet'])
    clean_train_tweets = clean(train['Tweet'])
    clean_test_tweets = clean(test['Tweet'])

    all_clean_tweets = clean_train_tweets + clean_test_tweets

    wordfreq = {}
    for sentence in all_clean_tweets:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    most_freq = heapq.nlargest(sum(wordfreq.values()), wordfreq, key=wordfreq.get)

    train_sentence_vectors = []
    for sentence in clean_train_tweets:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        train_sentence_vectors.append(sent_vec)

    test_sentence_vectors = []
    for sentence in clean_test_tweets:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        test_sentence_vectors.append(sent_vec)

    train_sentence_vectors = np.asarray(train_sentence_vectors)
    test_sentence_vectors = np.asarray(test_sentence_vectors)

    len_vocab = len(wordfreq.keys())
    print(len_vocab)

    train_mat = np.matrix(train_sentence_vectors)
    with open('graduate-project-dalcantara7/train_matrix.txt', 'w+') as f:
        for line in train_mat:
            np.savetxt(f, line, fmt='%1.0f')

    test_mat = np.matrix(test_sentence_vectors)
    with open('graduate-project-dalcantara7/test_matrix.txt', 'w+') as f:
        for line in test_mat:
            np.savetxt(f, line, fmt='%1.0f')

    return train_sentence_vectors, test_sentence_vectors, len_vocab

    #use bidirectional RNN for model

def clean(tweet_set):
    to_return = []

    for tweet in tweet_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        to_return.append(sans_extra_spaces)
        # array_words = sans_extra_spaces.split()
        # to_return += [word for word in array_words if word not in set(stopwords.words('english'))]
        # print(to_return)
    
    return to_return

if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="graduate-project-dalcantara7/2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="graduate-project-dalcantara7/2018-E-c-En-train.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    train_word_matrix, test_word_matrix, vocab_len = preprocess_data(train_data, test_data)

    # makes predictions on the dev set
    train_matrix = np.loadtxt("graduate-project-dalcantara7/train_matrix.txt", dtype='i', delimiter=' ')
    test_matrix = np.loadtxt("graduate-project-dalcantara7/test_matrix.txt", dtype='i', delimiter=' ')
    vocab_len = 16052
    test_predictions = train_and_predict(train_matrix, test_matrix, np.asarray(train_data.iloc[:,2:11]), np.asarray(test_data.iloc[:,2:11]), vocab_len)

    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))
