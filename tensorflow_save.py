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
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl


x = load_model("model.h5")
print ([layer.name for layer in x.layers])

# word2vec = load_model("w2v.h5")

word2vec = Model(inputs=x.input[0], output=x.get_layer("embedding").output)
del x

with K.get_session() as sess:
#     tf.saved_model.simple_save(
#         sess,
#         "models/1",
#         inputs={'input_words': word2vec.input},
#         outputs={t.name:t for t in word2vec.outputs}
#     )

# print (word2vec.input)
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"input_words": word2vec.input}, {"prediction":word2vec.output})

    valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
    if(valid_prediction_signature == False):
        raise ValueError("Error: Prediction signature not valid!")
        
    builder = saved_model_builder.SavedModelBuilder('models/2')
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()