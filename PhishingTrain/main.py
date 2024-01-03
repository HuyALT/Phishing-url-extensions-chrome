import numpy as np
import warnings

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    balanced_df = pd.read_csv('Dataset/modified_dataset.csv')

    # Separate URLs and labels in the dataset
    URLs = balanced_df['URL']
    labels = balanced_df['Label']

    longest_url_index = balanced_df['URL'].apply(len).idxmax()
    shortest_url_index = balanced_df['URL'].apply(len).idxmin()

    # Get the longest and shortest URLs
    longest_url = balanced_df.loc[longest_url_index, 'URL']
    shortest_url = balanced_df.loc[shortest_url_index, 'URL']

    mapping = {'bad': 0, 'good': 1}
    balanced_df['Label'] = balanced_df['Label'].map(mapping)

    balanced_df["char_length"] = balanced_df["URL"].apply(len)

    quant_99 = balanced_df.char_length.quantile(0.99)

    final_df = balanced_df[balanced_df["char_length"] <= quant_99]
    final_df = final_df[final_df["char_length"] > 14.0]

    train_df, test_df = train_test_split(final_df, test_size=0.2, stratify=final_df['Label'], random_state=42)

    train_df = train_df[['URL', 'Label']]
    test_df = test_df[['URL', 'Label']]

    X_train = train_df['URL']
    y_train = train_df['Label']
    X_test = test_df['URL']
    y_test = test_df['Label']

    corpus_train = []  # Initialize an empty list called corpus_train
    corpus_test = []  # Initialize an empty list called corpus_test
    sentence = []  # Initialize an empty list called sentence

    for ur in X_train:
        corpus_train.append(ur)  # Append the current 'ur' to the corpus_train list
        ur = (" ").join(ur)  # Join the elements of 'ur' with a space and assign it back to 'ur'
        sentence.append(ur)  # Append the modified 'ur' to the sentence list

    # Iterate over each element 'ur' in the list 'X_test'
    for ur in X_test:
        corpus_test.append(ur)  # Append the current 'ur' to the corpus_test list
        ur = (" ").join(ur)  # Join the elements of 'ur' with a space and assign it back to 'ur'
        sentence.append(ur)  # Append the modified 'ur' to the sentence list

    # Convert each string in corpus_train to lowercase and store the results in corpus_train
    corpus_train = [string.lower() for string in corpus_train]

    # Convert each string in corpus_test to lowercase and store the results in corpus_test
    corpus_test = [string.lower() for string in corpus_test]

    combined_text = " ".join(sentence)
    words = combined_text.split()
    unique_words = set(words)
    voc_size = len(unique_words) + 1

    corpus_mixed = corpus_train + corpus_test

    tokenizer = Tokenizer(char_level=True, num_words=voc_size)
    tokenizer.fit_on_texts(corpus_mixed)

    tokenized_train = tokenizer.texts_to_sequences(corpus_train)
    tokenized_test = tokenizer.texts_to_sequences(corpus_test)

    sent_len = max(len(sen) for sen in tokenized_train)


    embedded_docs_train = pad_sequences(tokenized_train, padding='post', maxlen=sent_len)
    embedded_docs_test = pad_sequences(tokenized_test, padding='post', maxlen=sent_len)

    X_train = np.array(embedded_docs_train)
    y_train = np.array(y_train)

    X_test = np.array(embedded_docs_test)
    y_test = np.array(y_test)

    embedding_vec_feature = 64

    model = Sequential()
    model.add(Embedding(voc_size, embedding_vec_feature, input_length=sent_len))
    model.add(Conv1D(filters=256, kernel_size=5, activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=4, strides=2, padding='valid'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=8, strides=1, padding='valid'))
    # model.add(Conv1D(filters=8, kernel_size=32, activation='relu',strides=1))
    # model.add(MaxPooling1D(pool_size=1,strides=1,padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for layer in model.layers:
        print(layer.name, "Input Shape:", layer.input_shape)
    model.summary()

    history = model.fit(X_train, y_train, epochs=15, batch_size=45, validation_split=0.1, )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(loss)

    model.save('Model/cnn.h5')

    import pickle

    # Save the tokenizer to a file
    with open('Model/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    with open('Model/sent_len.txt', 'w') as f:
        f.write(str(sent_len))
