from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine
# from keras.models import load_model, Model 
from scipy import linalg, mat, dot
import numpy as np
import flask
import io
from flask import request
import json
import pickle 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unicodedata
import math
import string
import requests
import sys
import tensorflow as tf 
from commonregex import CommonRegex

K = tf.keras.backend
load_model = tf.keras.models.load_model

stop_words = set(stopwords.words('english'))

w2v = pickle.load(open("w2v.bin", "rb"))
# v2w = pickle.load(open("v2w.bin", "rb"))

model = load_model("model.h5")
word2vec = tf.keras.models.Model(inputs=model.input[0], outputs=model.get_layer("embedding").output)
word2vec._make_predict_function()


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

def create_vec_job(data):
    data = clean([data])[0]
    data = [d.split(" ") for d in data][0]
    data = [subword(w, w2v) for w in data]
    data = pad_sequences(data, maxlen=25, value=len(w2v), padding="post")

    x = word2vec.predict(data)

    for i in range(len(x)):
        a = x[i]
        a_d = np.linalg.norm(a)
        a = a / a_d
        x[i] = a
    x = sum(x)/len(x)
    return x

def create_vec_resume(data):
    parse = CommonRegex(data)

    for i in parse.phones:
        data = data.replace(i, "")

    for i in parse.emails:
        data = data.replace(i, "")

    for i in parse.street_addresses:
        data = data.replace(i, "")

    data = sent_tokenize(data)
    data = clean(data)[0]

    nd0 = []
    for i in range(0, len(data) - 5):
        nd0.append(data[i:i + 5])

    print (np.array(nd0).shape)


    data = nd0 
    data = [[d.split(" ") for d in a] for a in data]
    data = [[[subword(w, w2v) for w in d] for d in a] for a in data]
    data = [[pad_sequences(d, maxlen=25, value=len(w2v), padding="post") for d in a] for a in data]


    rets = []
    for z in data:
        for b in z:
            x = word2vec.predict(b)
            for i in range(len(x)):
                a = x[i]
                a_d = np.linalg.norm(a)
                a = a / a_d
                x[i] = a

            rets.append(sum(x)/len(x))
    return rets

def get_avg_n(n, sims):
    sims_ = sorted(sims)[::-1]
    sims_ = sims_[:n]
    print (sims_)
    return (sum(sims_)/len(sims_))

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

@app.route("/predict-employer", methods=["POST"])
def predict_employer():

    data = {"success": False}

    if flask.request.method == "POST":
        x = compare(request.form["data1"].lower(), request.form["data2"].lower())
        # data1 = create_vec(request.form["data1"], resume=True)
        # data2 = create_vec(request.form["data2"])
        
    return flask.jsonify(x)

@app.route("/predict-user", methods=["POST"])
def predict_user():

    data = {"success": False}

    if flask.request.method == "POST":
        # x = compare(request.form["data1"], request.form["data2"])
        data1 = create_vec_job(request.form["data1"].lower())
        data2 = create_vec_job(request.form["data2"].lower())
        
    return flask.jsonify(calculate_similarity(data1, data2))

def compare(d1, d2):
    d2_ret = create_vec_job(d2)
    d1_rets = create_vec_resume(d1)

    sims = []
    for i in d1_rets:
        # print (i)
        sims.append(calculate_similarity(i, d2_ret))

    x = get_avg_n(int(len(sims)*0.35), sims)
    print(x)
    # x = 1.011951 + (0.0001021114 - 1.011951)/(1 + (x/0.5718397)**5.644143)
    return x

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # load_model()
    app.run(port=5000, host='0.0.0.0')
