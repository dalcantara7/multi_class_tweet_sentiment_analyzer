from nltk.stem.snowball import SnowballStemmer
from keras import models, layers
from keras.preprocessing.text import Tokenizer
from keras import callbacks
import keras.preprocessing.sequence as kps
import keras
import pandas as pd
import numpy as np
import argparse
import zipfile
import sklearn.metrics
import json
import re
import emoji

emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

NUM_WORDS = 120000


def train_and_predict(train_word_data, train_labels,
                      test_word_data, test_labels, vocab_len, full_test_data, embedding_layer):

    model, kwargs = create_model(vocab_len, embedding_layer)

    kwargs.update(x=train_word_data, y=train_labels,
                  epochs=2, validation_data=(test_word_data, test_labels))

    model.fit(**kwargs)

    dev_predictions = full_test_data.copy()

    predictions = model.predict(test_word_data)

    predictions = np.where(predictions > 0.4, 1, 0)

    dev_predictions[emotions] = predictions

    return dev_predictions

def create_model(vocab_len, embedding_layer):
    model = models.Sequential()
    model.add(embedding_layer)
    model.add(layers.Bidirectional(layers.GRU(500)))
    model.add(layers.Dense(325, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(11, activation ='sigmoid'))

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    return [model, {'batch_size' : 16, 'callbacks' : [callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]}] # callback only used here to restore best weights if need be

def preprocess_data(train, test):
    EMBEDDING_DIM = 200
    clean_train = []
    clean_test = []

    with open('graduate-project-dalcantara7/clean_train_tweets.txt', 'r') as f:
        for tweet in f:
            clean_train.append(tweet.rstrip())

    with open('graduate-project-dalcantara7/clean_test_tweets.txt', 'r') as f:
        for tweet in f:
            clean_test.append(tweet.rstrip())

    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(clean_train)
    
    # convert to to token sequences of equal length
    train_tweet_indices = tokenizer.texts_to_sequences(clean_train)
    word_index = tokenizer.word_index
    train_tweet_indices = kps.pad_sequences(train_tweet_indices, padding='post')

    test_tweet_indices = tokenizer.texts_to_sequences(clean_test)
    test_tweet_indices = kps.pad_sequences(test_tweet_indices, padding='post', maxlen=len(train_tweet_indices[0]))

    # handling of glove in Keras embedding layer
    embeddings_index = {}
    f = open("graduate-project-dalcantara7/glove.twitter.27B/glove.twitter.27B.200d.txt")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=len(train_tweet_indices[0]),
                            trainable=True,
                            mask_zero=True)

    return train_tweet_indices, test_tweet_indices, embedding_layer

def clean_tweets(train_set, test_set):
    clean_train = []
    clean_test = []
    sno = SnowballStemmer('english')

    # converts emojis to texts, lowers, removes special characters including '_', stems words
    for tweet in train_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        sans_underscore = sans_extra_spaces.replace('_', ' ')
        split = sans_underscore.split()
        stemmed = ' '.join([sno.stem(word) for word in split])
        clean_train.append(stemmed)

    for tweet in test_set:
        demojized = emoji.demojize(tweet)
        lowered = demojized.lower()
        sans_special_chars = re.sub(r'\W',' ', lowered)
        sans_extra_spaces = re.sub(r'\s+',' ', sans_special_chars)
        sans_underscore = sans_extra_spaces.replace('_', ' ')
        split = sans_underscore.split()
        stemmed = ' '.join([sno.stem(word) for word in split])
        clean_test.append(stemmed)

    with open('graduate-project-dalcantara7/clean_train_tweets.txt', 'w+') as f:
        for tweet in clean_train:
            f.write("%s\n" % tweet)

    with open('graduate-project-dalcantara7/clean_test_tweets.txt', 'w+') as f:
        for tweet in clean_test:
            f.write("%s\n" % tweet)

if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="graduate-project-dalcantara7/2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="graduate-project-dalcantara7/2018-E-c-En-test.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    clean_tweets(train_data['Tweet'], test_data['Tweet'])

    train_word_indices, test_word_indices, embedding_layer = preprocess_data(train_data, test_data)

    train_labels = np.asarray(train_data.iloc[:,2:13]) # get labels
    test_labels = np.asarray(test_data.iloc[:,2:13]) # get labels

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_word_indices, train_labels, test_word_indices, test_labels, NUM_WORDS, test_data, embedding_layer)

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))

    # saves predictions and creates submission zip file
    test_predictions.to_csv("graduate-project-dalcantara7/E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('graduate-project-dalcantara7/submission.zip', mode='w') as submission_zip:
        submission_zip.write("graduate-project-dalcantara7/E-C_en_pred.txt")
