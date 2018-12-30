from keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
import data_processor
import pickle 
import numpy as np
from keras import backend as K
import tensorflow as tf 


x = load_model("model.h5")
print ([layer.name for layer in x.layers])

# word2vec = load_model("w2v.h5")

word2vec = Model(inputs=x.input[0], output=x.get_layer("embedding").output)
del x

with K.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        "8",
        inputs={'input_words': word2vec.input},
        outputs={t.name:t for t in word2vec.outputs}
    )