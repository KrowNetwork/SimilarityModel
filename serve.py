from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine
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


stop_words = set(stopwords.words('english'))

w2v = pickle.load(open("w2v.bin", "rb"))
v2w = pickle.load(open("v2w.bin", "rb"))

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
            w.append(len(w2v) - 1)

    return w

def subword_embedding(string):
    # w2s = {}
    # s2c = {}

    s = "<" + string + ">"
    arr = []
    for a in range(len(s) - 2):
        arr.append(s[a:a+3])
    # for a in range(len(a)):

    return arr

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def create_vec(data):
    data = clean([data])[0]
    data = [d.split(" ") for d in data][0]
    data = [subword(w, w2v) for w in data]
    print (w2v)
   
    data = pad_sequences(data, maxlen=25, value=11411, padding="post")
    # print (data[0])
    payload = {
        "signature_name":"serving_default",
        "instances": [
            {"input_words": data.tolist()}
        ]
    }
    r = requests.post('http://localhost:9000/v1/models/2:predict', json=payload)
    r = requests.post('http://localhost:9000/v1/models/2:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    print (r.text)
    x = (r.json()["predictions"])
    print (x)
    x = np.array(x)
    print (x.shape)
   
    x = sum(np.array(x[0]))
   
    pred = json.loads(r.content.decode('utf-8'))
    return x

def calculate_similarity(v1, v2):
    # a = mat(v1)
    # b = mat(v2)
    # return dot(a,b.T)/linalg.norm(a)/linalg.norm(b)
    print (v1)
    print (v2)
    sim = 1 - cosine(v1, v2)
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
