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

# with K.get_session() as sess:
#     tf.saved_model.simple_save(
#         sess,
#         "models/1",
#         inputs={'input_words': word2vec.input},
#         outputs={t.name:t for t in word2vec.outputs}
#     )

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import     build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

export_path = 'model/2'
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'input_words': word2vec.input},
                                  outputs={'embedding': word2vec.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()