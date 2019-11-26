from nltk.corpus import stopwords
from keras import models, layers
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence as kps
import pandas as pd
import numpy as np
import argparse
import zipfile
import sklearn.metrics
import json
import re
import emoji
import keras


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}


def train_and_predict(train_word_data, train_labels,
                      test_word_data, test_labels, vocab_len, full_test_data):

    # doesn't train anything; just predicts 1 for all of dev set
    # dev_predictions = dev_data.copy()
    # dev_predictions[emotions] = 1

    model, kwargs = create_model(vocab_len)

    kwargs.update(x=train_word_data, y=train_labels,
                  epochs=13, validation_data=(test_word_data, test_labels))

    model.fit(**kwargs)

    dev_predictions = full_test_data.copy()

    predictions = model.predict(test_word_data)
    # mat = np.matrix(predictions)
    # with open('graduate-project-dalcantara7/predictions_continuous.txt', 'w+') as f:
    #     for line in mat:
    #         np.savetxt(f, line, fmt='%.2f')

    predictions = np.round(predictions)
    # mat = np.matrix(predictions)
    # with open('graduate-project-dalcantara7/predictions.txt', 'w+') as f:
    #     for line in mat:
    #         np.savetxt(f, line, fmt='%1.0f')

    dev_predictions[emotions] = predictions
    # print(dev_predictions)

    return dev_predictions

def create_model(vocab_len):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_len, 200, mask_zero=True))
    model.add(layers.Bidirectional(layers.GRU(64)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(11, activation ='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return [model, {}]

def preprocess_data(train, test):
    clean_train = []
    clean_test = []

    with open('graduate-project-dalcantara7/clean_train_tweets.txt', 'r') as f:
        for tweet in f:
            clean_train.append(tweet.rstrip())

    with open('graduate-project-dalcantara7/clean_test_tweets.txt', 'r') as f:
        for tweet in f:
            clean_test.append(tweet.rstrip())

    tokenizer = Tokenizer(num_words=120000)
    tokenizer.fit_on_texts(clean_train)
    
    train_tweet_indices = tokenizer.texts_to_sequences(clean_train)
    train_tweet_indices = kps.pad_sequences(train_tweet_indices, padding='post')

    test_tweet_indices = tokenizer.texts_to_sequences(clean_test)
    test_tweet_indices = kps.pad_sequences(test_tweet_indices, padding='post')

    return train_tweet_indices, test_tweet_indices

def clean_tweets(train_set, test_set):
    clean_train = []
    clean_test = []

    for tweet in train_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        sans_underscore = sans_extra_spaces.replace('_', ' ')
        split = sans_underscore.split()
        without_stop_words = ' '.join([word for word in split if word not in set(stopwords.words('english'))])
        clean_train.append(without_stop_words)

    for tweet in test_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        sans_underscore = sans_extra_spaces.replace('_', ' ')
        split = sans_underscore.split()
        without_stop_words = ' '.join([word for word in split if word not in set(stopwords.words('english'))])
        clean_test.append(without_stop_words)

    len_vocab = 0
    for tweet in clean_train:
        len_vocab += len(tweet)

    with open('graduate-project-dalcantara7/clean_train_tweets.txt', 'w+') as f:
        for tweet in clean_train:
            f.write("%s\n" % tweet)

    with open('graduate-project-dalcantara7/clean_test_tweets.txt', 'w+') as f:
        for tweet in clean_test:
            f.write("%s\n" % tweet)

    # return clean_train, clean_test, len_vocab

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

    # clean_tweets(train_data['Tweet'], test_data['Tweet'])

    train_word_indices, test_word_indices = preprocess_data(train_data, test_data)

    train_labels = np.asarray(train_data.iloc[:,2:13])
    test_labels = np.asarray(test_data.iloc[:,2:13])

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_word_indices, train_labels, test_word_indices, test_labels, 120000, test_data)

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))

    # saves predictions and creates submission zip file
    test_predictions.to_csv("graduate-project-dalcantara7/E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('graduate-project-dalcantara7/submission.zip', mode='w') as submission_zip:
        submission_zip.write("graduate-project-dalcantara7/E-C_en_pred.txt")
