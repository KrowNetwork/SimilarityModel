from keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
import data_processor
import pickle 
import numpy as np
from keras import backend as K
import tensorflow as tf 
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants




K.set_learning_phase(0)

x = load_model("model.h5")
print ([layer.name for layer in x.layers])

# word2vec = load_model("w2v.h5")

keras_model = Model(inputs=x.input[0], output=x.get_layer("embedding").output)
del x

# Load the Keras model
# keras_model = load_model(path_to_h5)

# Build the Protocol Buffer SavedModel at 'export_path'
builder = saved_model_builder.SavedModelBuilder("models/2")

# Create prediction signature to be used by TensorFlow Serving Predict API
signature = predict_signature_def(inputs={"input_words": keras_model.input},
                                    outputs={"embeddings": keras_model.output})

with K.get_session() as sess:
    # Save the meta graph and the variables
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                        signature_def_map={"predict": signature})

builder.save()