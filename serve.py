from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine
# from keras.model import load_model, Model 
from scipy import linalg, mat, dot
import numpy as np
import flask
import io
from flask import request
import json
import pickle 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unicodedata
import math
import string
import requests
import sys

stop_words = set(stopwords.words('english'))

w2v = pickle.load(open("w2v.bin", "rb"))
# v2w = pickle.load(open("v2w.bin", "rb"))


def clean(docs):
    docs2 = []
    vocab = []
    for i in docs:
        words2 = []
        tokens = word_tokenize(i)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        for w in words:
            if w not in stop_words:
                w = unicodedata.normalize('NFD', w)
                w = u"".join([c for c in w if not unicodedata.combining(c)])
                words2.append(w)
        words = words2
        # words = [porter.stem(word) for word in words]
        vocab.extend(words)
        docs2.append(" ".join(words))

    return docs2, list(set(vocab))

def subword(x, w2v):
    w = []
    # print (x)
    for a in subword_embedding(x):
        try:
            w.append(w2v[a])
        except:
            w.append(len(w2v))

    return w

def subword_embedding(string):

    s = "<" + string + ">"
    arr = []
    for a in range(len(s) - 2):
        arr.append(s[a:a+3])

    return arr

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def create_vec(data):
    data = clean([data])[0]
    data = [d.split(" ") for d in data][0]
    data = [subword(w, w2v) for w in data]
    data = pad_sequences(data, maxlen=25, value=len(w2v), padding="post")

    payload = {
        "signature_name":"serving_default",
        "instances": [
            {"input_words": data.tolist()}
        ]
    }

    r = requests.post('http://localhost:9000/v1/models/%s:predict' % sys.argv[1], json=payload)

    x = (r.json()["predictions"])
    x = np.array(x)
    for i in range(len(x)):
        a = x[i]
        a_d = np.linalg.norm(a)
        a = a / a_d
        x[i] = a
    x = sum(x)/len(x)
    print (x)
    # x = sum(np.array(x[0]))
    pred = json.loads(r.content.decode('utf-8'))
    return x

def calculate_similarity(v1, v2):
    sim = 1 - cosine(v1, v2)
    # print (sim)
    # sim = (0.25 * sim) + (1.25 * (sim ** 2))
    sim = 5.5511150000000004e-17 + 0.375*sim + 0.9375*sim**2
    if sim > 1:
        sim = 0.99
    if sim < 0:
        sim = 0.01
    # sim = (sim - -1)/ (1 - -1)
    # print (type(sim))
    if sim == np.nan:
        sim = 0
    return sim

@app.route("/predict", methods=["POST"])
def predict():

    data = {"success": False}

    if flask.request.method == "POST":
        data1 = create_vec(request.form["data1"])
        data2 = create_vec(request.form["data2"])
        
    return flask.jsonify(calculate_similarity(data1, data2))

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # load_model()
    app.run(port=5000, host='0.0.0.0')
