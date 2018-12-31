from comet_ml import Experiment
import pickle
import time
import numpy as np
import sys
import os
import psutil
import copy
import tensorflow as tf
process = psutil.Process(os.getpid())
import data_processor
import keras
from keras.layers import Input, Dense, LSTM, Flatten, Conv1D, Reshape, Embedding, Lambda, Dot, Lambda, Add
from keras.models import Model
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from keras.layers import Layer 
import keras.backend as K
import random
from keras.preprocessing.sequence import skipgrams
from trainer import ModelTrainer
from model import create_model

# tf.enable_eager_execution()
batch_size = 2048 * 2
epochs = 7500
learning_rate = 0.0001
dataset_size = 200000
emb_dim = 500
gram_size = 3
context_size = 3
context_type = "before"
input_size = 25

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.95

session = tf.InteractiveSession(config=config)

K.tensorflow_backend.set_session(session)


experiment = Experiment(api_key="X2EWOW2J9duilIJlb4TaRjWbO",
                        project_name="character-based-word2vec", workspace="tgs266")


def sum_embedding(x):
    return K.sum(x, axis=1, keepdims=False)
def sum_embedding_0(x):
    return K.sum(x, axis=2, keepdims=False)

def subword_embedding_fct(x):
    w = []
    # print (x)
    for a in data_processor.subword_embedding(x):
        w.append(w2v[a])

    return w

# def sum_embedding(x):
    

def mse(y, pred):
    # rounded = tf.math.round(pred)
    # rounded = 
    loss = tf.reduce_mean(tf.squared_difference(y, pred))
    return loss

def acc_round(y, pred):
    rounded = tf.math.round(pred)
    acc = tf.reduce_mean((y - pred)/y * 100)
    print (acc)

    return acc

def randomize(l1, l2, l3):
    l = list(range(len(l1)))
    random.shuffle(l)
    l4, l5, l6 = [], [], []
    for i in l:
        l4.append(l1[i])
        l5.append(l2[i])
        l6.append(l3[i])

    return l4, l5, l6




documents = pickle.load(open("clean.bin", "rb"))
w2v = pickle.load(open("w2v.bin", "rb"))
v2w = pickle.load(open("v2w.bin", "rb"))




words = []
context = []
# ins = [[], []]
words, context = data_processor.create_dataset(documents, w2v, gram_size, context_type=context_type)

print (len(words))
full_dataset_size = len(words)
words, context, bl = randomize(words, context, [1] * len(words))
words = words[:dataset_size]
context = context[:dataset_size]

vocab_size = len(w2v)
print (vocab_size)
# exit()

words_, context_ = words[int(0.5 * len(words)):], context[int(0.5 * len(words)):]
labels_neg = [0] * len(words_)
random.shuffle(words_)

words, context = words[:int(0.5 * len(words))], context[:int(0.5 * len(words))]
lables = [1] * len(words)

words.extend(words_)
context.extend(context_)
lables.extend(labels_neg)
# print (words[0])
# print (context[0])
# exit()

words, context, lables = randomize(words, context, lables)

words_ = []
context_ = []
labels_neg = []

params={
    "batch_size":batch_size,
    "epochs":epochs,
    "learning_rate":learning_rate,
    "char_vocab_size": vocab_size,
    "dataset_size": dataset_size,
    "full_dataset_size": full_dataset_size,
    "embedding_dimsension": emb_dim,
    "gram_size": gram_size,
    "context_size": context_size,
    "context_type": context_type,
    "input_size": input_size}

experiment.log_parameters(params)

model = create_model(input_size, context_size, emb_dim, vocab_size, learning_rate, verbose=True)
# del _

# exit()
model_trainer = ModelTrainer(K, model, None, words, context, lables, vocab_size, experiment, batch_size=batch_size, shuffle=True)
model_trainer.train(epochs)

# xy = "computers technology"
# zy = "technology art"
# x = "computers"
# y = "technology"
# z = "art"


# vecs = data_processor.create_vecs(data_processor.clean([x, y, z])[0])
# # print (vecs)

# print ("%s vs %s: %s" % (x, y, model.predict([vecs[0], vecs[1]])))
# print ("%s vs %s: %s" % (x, z, model.predict([vecs[0], vecs[2]])))
# a = word2vec.predict([vecs[0]])
# b = word2vec.predict([vecs[1]])
# # print (vecs)
# print (sim_model.predict([vecs[0], vecs[1]]))
# print (sim_model.predict([vecs[0], vecs[2]]))

# # print (len(a))
# # print (len(a[0]))
# # print (len(a[0][0]))
# # print (a)
# # print (a[0][-1])
# # print (a[0][-2])
# # print (" ")
# print (a[0][1], vecs[0][0][1])
# print (b[0][5], vecs[1][0][5])

model.save("model.h5")
word2vec.save("w2v.h5")
sim_model.save("sim.h5")