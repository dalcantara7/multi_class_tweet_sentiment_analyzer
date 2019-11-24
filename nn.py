import nltk 
from nltk.corpus import stopwords
from keras import models, layers
import heapq
# import glove
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as kpt
import argparse
import zipfile
import sklearn.metrics
import pandas as pd
import numpy as np
import json
import re
import emoji
import string
import keras
# import sys
# np.set_printoptions(threshold=sys.maxsize)


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}


def train_and_predict(train_word_data, train_labels,
                      test_word_data, test_labels, vocab_len):

    # doesn't train anything; just predicts 1 for all of dev set
    # dev_predictions = dev_data.copy()
    # dev_predictions[emotions] = 1

    model, kwargs = create_model(vocab_len)
    # print(len(train_word_data))
    # print(len(test_word_data))
    print(train_labels)
    print(len(test_labels))

    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)

    kwargs.update(x=train_word_data, y=train_labels,
                  epochs=10, validation_data=(test_word_data, test_labels))

    model.fit(**kwargs)

    return model.predict(test_word_data, test_labels)

def create_model(vocab_len):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_len, 200, mask_zero=True))
    model.add(layers.Bidirectional(layers.GRU(64, return_sequences=True)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation ='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return [model, {}]

def preprocess_data(train, test):

    clean_train, clean_test, len_vocab = clean_tweets(train['Tweet'], test['Tweet'])

    tokenizer = Tokenizer(num_words=len_vocab)
    tokenizer.fit_on_texts(clean_train)
    dictionary = tokenizer.word_index
    with open('graduate-project-dalcantara7/dictionary.json', 'w+') as dictionary_file:
        json.dump(dictionary, dictionary_file)

    tokenizer.fit_on_texts(clean_test)
    dictionary = tokenizer.word_index
    with open('graduate-project-dalcantara7/dictionary.json', 'a') as dictionary_file:
        json.dump(dictionary, dictionary_file)
    
    train_tweet_indices = []
    for tweet in clean_train:
        word_indices = convert_tweet_to_index_array(dictionary, tweet)
        train_tweet_indices.append(word_indices)

    train_tweet_indices = np.asarray(train_tweet_indices)

    test_tweet_indices = []
    for tweet in clean_test:
        word_indices = convert_tweet_to_index_array(dictionary, tweet)
        test_tweet_indices.append(word_indices)

    test_tweet_indices = np.asarray(test_tweet_indices)

    train_tweet_matrix = tokenizer.sequences_to_matrix(train_tweet_indices, mode='binary')
    test_tweet_matrix = tokenizer.sequences_to_matrix(test_tweet_indices, mode='binary')

    return train_tweet_matrix, test_tweet_matrix, len_vocab

def convert_tweet_to_index_array(dictionary, tweet):
    return [dictionary[word] for word in kpt.text_to_word_sequence(tweet)]

def clean_tweets(train_set, test_set):
    clean_train = []
    clean_test = []

    for tweet in train_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        split = sans_extra_spaces.split()
        without_stop_words = ' '.join([word for word in split if word not in set(stopwords.words('english'))])
        # vocabulary += [word for word in clean_train]
        clean_train.append(without_stop_words)

    for tweet in test_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        split = sans_extra_spaces.split()
        without_stop_words = ' '.join([word for word in split if word not in set(stopwords.words('english'))])
        # vocabulary += [word for word in clean_train]
        clean_test.append(without_stop_words)

    len_vocab = 0
    for tweet in clean_train:
        len_vocab += len(tweet)

    for tweet in clean_test:
        len_vocab += len(tweet)

    return clean_train, clean_test, len_vocab

if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="graduate-project-dalcantara7/2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="graduate-project-dalcantara7/2018-E-c-En-dev.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)
    # print(train_data.iloc[:,2:11][0])

    train_word_indices, test_word_indices, vocab_len = preprocess_data(train_data, test_data)

    train_labels = train_data.iloc[:,2:11].values
    test_labels = test_data.iloc[:,2:11].values

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_word_indices, test_word_indices, train_labels, test_labels, vocab_len)

    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))
