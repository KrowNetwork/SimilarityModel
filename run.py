

from comet_ml import Experiment

import pickle
import time
import numpy as np
import sys
import os
import tensorflow as tf
import psutil
import copy
# import tensorflow as tf
process = psutil.Process(os.getpid())
import data_processor
# import keras
# from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Conv1D, Reshape, Embedding, Lambda, Dot, Lambda, Add
# from tensorflow.keras import Model
# from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
# from tensorflow.keras.layers import Layer 
# import keras.backend as K
# from tensorflow.keras.optimizer_v2 import adam
import random
import tensorflow as tf
# from keras.preprocessing.sequence import skipgrams
from trainer import ModelTrainer
from model import create_model
import h5py 
from multiprocessing import freeze_support
# from tensorflow.contrib.optimizer_v2 import RMSPropOptimizer

import queue

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('load', False, 'If true, loads most recent model.h5')
flags.DEFINE_integer('starting_epoch', 0, 'Starting epoch for loaded model')

print (FLAGS.load)
# tf.enable_eager_execution()#device_policy=tf.contrib.eager.DEVICE_PLACEMENT_EXPLICIT)

tpu = False
if tpu:
    print ("RUNNING IN TPU MODE")
# with tf.device("/device:GPU:0"):
# print ("VERSION", tf.__version__)




# config = tf.ConfigProto()
# jit_level = tf.OptimizerOptions.ON_1
# config.graph_options.optimizer_options.global_jit_level = jit_level
if tpu:
    TPU_WORKER = 'grpc://' + "10.240.1.2:8470"
# config.allow_soft_placement=True
# config.log_device_placement=True
#     config.device_count = {'GPU': 0}

# sess = tf.InteractiveSession(config=config)

# K.set_session(sess)
# tf.enable_eager_execution()
# tf.logging.set_verbosity(tf.logging.INFO)





  
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

def generate(swords, scontext, dataset_size):
    words, context, bl = randomize(swords, scontext, [1] * len(swords))
    words = words[:dataset_size]
    context = context[:dataset_size]

    words_, context_ = words[int(0.5 * len(words)):], context[int(0.5 * len(words)):]
    labels_neg = [0] * len(words_)
    random.shuffle(words_)

    words, context = words[:int(0.5 * len(words))], context[:int(0.5 * len(words))]
    labels = [1] * len(words)

    words.extend(words_)
    context.extend(context_)
    labels.extend(labels_neg)
    # print (words[0])
    # print (context[0])
    # exit()

    words, context, labels = randomize(words, context, labels)

    words_ = []
    context_ = []
    labels_neg = []

    return words, context, labels



# if __name__ == "__main__":
#     freeze_support()
batch_size = 2048 * 4
epochs = 2500
learning_rate = 0.0001
dataset_size = 128**3
emb_dim = 300
gram_size = 3
context_size = 3
context_type = "before"
input_size = 25

# if int(datasetbatch_size)

buffer_size = int(dataset_size/batch_size)
# q = queue.Queue(buffer_size)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# # # config.allow_soft_placement=True
# # # jit_level = tf.OptimizerOptions.ON_1
# # # config.graph_options.optimizer_options.global_jit_level = jit_level
# config.gpu_options.per_process_gpu_memory_fraction = 0.95

# session = tf.InteractiveSession(config=config)

# K.set_session(session)
K = tf.keras.backend
# K.set_session(tf.Session(config=config))
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

print ("Using " + str(K.floatx()))


experiment = Experiment(api_key="X2EWOW2J9duilIJlb4TaRjWbO",
                        project_name="character-based-word2vec", workspace="tgs266")

documents = pickle.load(open("clean.bin", "rb"))
w2v = pickle.load(open("w2v.bin", "rb"))
# v2w = pickle.load(open("v2w.bin", "rb"))
# swords = pickle.load(open("data/words.bin", "rb"))
# scontext = pickle.load(open("data/context.bin", "rb"))
# swords = np.load("words.bn.npz")["words"]
# scontext = np.load("context.bin.npz")["context"]
print ("Files downloaded")
# words = []
# context = []
# ins = [[], []]


# print (len(words))
# full_dataset_size = len(words)
vocab_size = len(w2v)


# reload_model = False
params={
    "batch_size":batch_size,
    "epochs":epochs,
    "learning_rate":learning_rate,
    "char_vocab_size": vocab_size,
    "dataset_size": dataset_size,
    # "full_dataset_size": full_dataset_size,
    "embedding_dimsension": emb_dim,
    "gram_size": gram_size,
    "context_size": context_size,
    "context_type": context_type,
    "input_size": input_size}

experiment.log_parameters(params)
if FLAGS.load == False:
    model = create_model(input_size, context_size, emb_dim, vocab_size, learning_rate, batch_size, verbose=True)
else:
    model = tf.keras.models.load_model("model.h5")
# model.save("model.h5")
# model = keras.models.load_model("model.h5")
if tpu:
    model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
    model.compile(
    optimizer = tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(learning_rate)),  #tf.keras.optimizers.RMSprop(lr=learning_rate),
    loss="binary_crossentropy",
    metrics=['accuracy']
)
else:

    # with tf.device("GPU:0"):
    # tfopt = tf.contrib.optimizer_v2.RMSPropOptimizer(learning_rate)
    # tfopt = tf.contrib.opt.LazyAdamOptimizer(learning_rate)
    model.compile(
        # optimizer=tf.train.RMSPropOptimizer(learning_rate),
        # optimizer=tfopt,
        # optimizer=tf.keras.optimizers.SGD(lr=learning_rate),
                
        optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
        # loss="mse",
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
# # del _

# exit()

try:
    completed = []
    # for i in range(1500):
    # q = None
    random.shuffle(documents)
    words, context = data_processor.create_dataset(documents[:int(len(documents) * 0.35)], w2v, gram_size, context_type=context_type)   
    print (len(words)) 
    # print (np.array(words).shape)
    # print (words[:10])
    
    words, context, labels = generate(words, context, dataset_size)

    # p = ProducerThread(words, context, labels, batch_size, value, q)
    # print (words[:10])
    for a in words:
        completed.extend(a)
    completed = list(set(completed))
    print (len(completed))
    # lables = []
    q = queue.Queue(buffer_size)

    model_trainer = ModelTrainer(K, q, model, None, words, context, labels, vocab_size, experiment, batch_size=batch_size, shuffle=True)
    # if not tpu:
    model_trainer.train(0, epochs, start=FLAGS.starting_epoch, loaded=FLAGS.load)
    # model_trainer.generator_train(0, epochs)
    # else:
    #     model_trainer.tpu_train(0, epochs)
    print ("\nCovered %s" % (len(completed)))
    experiment.log_metric("covered", len(completed))
except KeyboardInterrupt:
    model.save("model.h5")
    exit()

    

    
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

