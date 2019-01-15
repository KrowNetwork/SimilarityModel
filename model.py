import pickle
import time
import numpy as np
import sys
import os
import psutil
import copy
process = psutil.Process(os.getpid())
import data_processor
# import keras
# from keras.layers import Input, Dense, LSTM, Flatten, Conv1D, Reshape, Embedding, Lambda, Dot, Lambda, Add, RepeatVector
# from keras import Model
# from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
# # from keras.layers import Layer 
# import keras.backend as K
import tensorflow as tf
import random
from keras.preprocessing.sequence import skipgrams
# from trainer import ModelTrainer
K = tf.keras.backend

def sum_embedding(x):
    return K.sum(x, axis=1, keepdims=False)
    # return tf.reduce_sum(x, axis=1)

def duplicate(x):
    return K.reshape(K.tile(x, [1,3]), (-1, 3, 25))

def avg(x):
#     x = K.sum(x, axis=1)
    x = K.sum(x, axis=2)
    x = K.mean(x, axis=1)
    # x = tf.reduce_sum(x, axis=2)
    # x = tf.reduce_mean(x, axis=1)
    # print ("a", x.shape)
    # print ("b", x.shape)
    return x
# sd = tf.keras.layers.Dense(300, activation="linear")
# def shared_dense(x):
#     return sd(x)

print (sum_embedding.__name__)
def create_model(input_size, context_size, emb_dim, vocab_size, learning_rate, batch_size, verbose=False):
#     with tf.device("/device:GPU:0"):


    input_target = tf.keras.layers.Input(batch_shape=(batch_size, input_size))
    # input_target_ = Lambda(duplicate)(input_target)
    input_target_ = tf.keras.layers.RepeatVector(3)(input_target)
    # input_target_ = input_target
    #     print (input_target.shape)
    input_context = tf.keras.layers.Input(batch_shape=(batch_size, context_size, input_size))

    embedding = tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=emb_dim, name="emb")


    target = embedding(input_target_)
    # target2 = Add()(target[0])
    print (target.shape) 

    #     target2 = Lambda(sum_embedding, name="c")(target)
    #     target2 = Lambda(sum_embedding)(target2)
    #     print (target2.shape) 
    target2 = tf.keras.layers.Lambda(avg)(target)
    print (target2.shape) 

    # context_ = Dense(30)(input_context)
    # context_ = Flatten()(context_)
    context_ = embedding(input_context)
    print (context_.shape)
    context_ = tf.keras.layers.Lambda(sum_embedding)(context_)
    context_ = tf.keras.layers.Lambda(sum_embedding)(context_)
    # context_ = Lambda(sum_embedding)(context_)
    # shared_dense = tf.keras.layers.Dense(emb_dim, activation="linear", name="embedding")
    # target2 = shared_dense(target2)
    # context_ = shared_dense(context_)

    # context_ = Reshape((256, ))(context_)
    # context_ = Lambda(sum_embedding)(context_)

    # dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = tf.keras.layers.Dot(1)([target2, context_])
    # dot_product = Flatten()(context_)
    # dot_product = context_

    similarity = tf.keras.layers.Dot(0)([target2, context_])

    # dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

    model = tf.keras.Model(inputs=[input_target, input_context], outputs=[output])
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    # model.compile(loss='binary_crossentropy', optimizer=tf.train.RMSPropOptimizer(learning_rate), metrics=["accuracy"])

    # model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])

    # word2vec = Model(input=input_target, output=target2)
    # sim_model = Model(input=[input_target, input_context], output=similarity)
    # model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])

    # model = keras.models.load_model("model.h5")
    if verbose:
        print (model.summary())
    return model

