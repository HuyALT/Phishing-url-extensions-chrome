import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
import lime
from lime.lime_text import LimeTextExplainer

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
from keras.models import load_model

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

@app.route('/api/urlcheck')
def Predict_data():
    with open('Model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('Model/cnn.h5')

    with open('Model/sent_len.txt','r') as f:
        sent_len = int(f.read())

    def lime_predict(URLs):
        sequences = tokenizer.texts_to_sequences(URLs)  # Use your Keras tokenizer to convert URLs to sequences
        padded_seqs = pad_sequences(sequences, maxlen=sent_len, padding='post', truncating='post')  # Pad the sequences
        predictions = model.predict(padded_seqs)  # Use your Keras model to predict probabilities
        return predictions

    explainer = LimeTextExplainer(class_names=['phishing', 'legitimate'],
                                  char_level=True)  # Use your actual class names here

    sample_url = request.args.get('url')

    explanation = explainer.explain_instance(sample_url, lime_predict, num_features=30, top_labels=1)

    if explanation.top_labels == [0]:
        data = {
            'status':'OK',
            'data':'phishing'
        }
        return jsonify(data)
    else:
        data = {
            'status':'OK',
            'data':'legitimate'
        }
        return jsonify(data)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)