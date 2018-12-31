import pickle
import time
import numpy as np
import sys
import os
import psutil
import copy
process = psutil.Process(os.getpid())
import data_processor
import keras
from keras.layers import Input, Dense, LSTM, Flatten, Conv1D, Reshape, Embedding, Lambda, Dot, Lambda, Add
from keras.models import Model
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from keras.layers import Layer 
import keras.backend as K
import tensorflow as tf
import random
from keras.preprocessing.sequence import skipgrams
# from trainer import ModelTrainer


def sum_embedding(x):
    return K.sum(x, axis=1, keepdims=False)

print (sum_embedding.__name__)
def create_model(input_size, context_size, emb_dim, vocab_size, learning_rate, verbose=False):

    input_target = Input((input_size, ))
    input_context = Input((context_size, input_size))

    embedding = Embedding(vocab_size + 1, emb_dim, name="emb")


    target = embedding(input_target)
    # target2 = Add()(target[0])
    target2 = Lambda(sum_embedding, name="embedding")(target)

    # context_ = Dense(30)(input_context)
    # context_ = Flatten()(context_)
    context_ = embedding(input_context)
    context_ = Lambda(sum_embedding)(context_)
    context_ = Lambda(sum_embedding)(context_)
    # context_ = Lambda(sum_embedding)(context_)


    # context_ = Reshape((256, ))(context_)
    # context_ = Lambda(sum_embedding)(context_)

    # dot_product = merge([target, context], mode='dot', dot_axes=1)
    dot_product = Dot(1)([target2, context_])
    # dot_product = Flatten()(context_)
    # dot_product = context_

    similarity = Dot(0)([target2, context_])

    # dot_product = Reshape((1,))(dot_product)
    # add the sigmoid output layer
    output = Dense(1, activation='sigmoid')(dot_product)

    model = Model(inputs=[input_target, input_context], outputs=output)
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=learning_rate), metrics=["accuracy"])

    # model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])

    # word2vec = Model(input=input_target, output=target2)
    # sim_model = Model(input=[input_target, input_context], output=similarity)
    # model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])
    if verbose:
        print (model.summary())
    return model
