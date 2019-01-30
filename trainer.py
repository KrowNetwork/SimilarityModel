import random
import numpy as np 
import keras
import progressbar
import sys
import time
from model import sum_embedding
from dataset_generator import Generator
from producer import ProducerThread
import os
import glob
import tensorflow as tf



class ModelTrainer():

    @staticmethod
    def randomize(l1, l2, l3):
        l = list(range(len(l1)))
        random.shuffle(l)
        l4, l5, l6 = [], [], []
        for i in l:
            l4.append(l1[i])
            l5.append(l2[i])
            l6.append(l3[i])

        return l4, l5, l6

    @staticmethod
    def pad(x, value):
        y = []
        if type(x[0][0]) != type(0):
            c_ = []
            for i in x:
                c_.append(list(keras.preprocessing.sequence.pad_sequences(i, maxlen=25, value=value, padding="post")))    
            y = c_
        else:
            y = list(keras.preprocessing.sequence.pad_sequences(x, maxlen=25, value=value, padding="post"))
        return y
    
    @staticmethod
    def format_eta(eta):
        if eta > 3600:
            eta_format = ('%d:%02d:%02d' %
                            (eta // 3600, (eta % 3600) // 60, eta % 60))
        elif eta > 60:
            eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
            eta_format = '%ds' % eta
        return eta_format

    def __init__(self, K, q, model, word2vec, words, context, labels, value, experiment, batch_size=2048, shuffle=False):
        self.model = model
        self.word2vec = word2vec
        self.words = words 
        self.context = context
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.value = value
        self.experiment = experiment
        self.K = K
        self.q = q

    def on_epoch_end(self):
        if self.shuffle == True:
            self.words, self.context, self.labels = self.randomize(self.words, self.context, self.labels)

    def shuffle_data(self):
        self.words, self.context, self.labels = self.randomize(self.words, self.context, self.labels)

    def tpu_train(self, z, epochs):
        files = glob.glob('trained/*')
        for f in files:
            os.remove(f)
        length = 30
        count = int(len(self.words)/self.batch_size)
        mod = int(count/length)
        # mod = int(np.floor(np.log10(len(self.words) / self.batch_size)) + 1) * 30
        print (mod)
        self.words = self.pad(self.words, self.value)
        self.context = self.pad(self.context, self.value)

        max_ = max(range(0, len(self.words), self.batch_size))
        max2 = len(range(0, len(self.words), self.batch_size))
        width = max2 % mod
        begin = time.time()
        
        self.model.fit([self.words, self.context], self.labels, epochs=epochs, batch_size=self.batch_size)
        # exit()
    
        self.model.save("trained/model.h5")

    def train(self, z, epochs, start=0, loaded=False):
        pct = 0.5
        files = glob.glob('trained/*')
        for f in files:
            os.remove(f)
        length = 30
        count = int((len(self.words) * pct)/self.batch_size)
        mod = int(count/length) + 1
        # mod = int(np.floor(np.log10(len(self.words) / self.batch_size)) + 1) * 30
        print (mod)
        self.words = self.pad(self.words, self.value)
        self.context = self.pad(self.context, self.value)

        max_ = max(range(0, int(len(self.words) * pct), self.batch_size))
        max2 = len(range(0, int(len(self.words) * pct), self.batch_size))
        # width = max2 % mod
        begin = time.time()
        
        
        # exit()
        epochs += start
        for e in range(start, epochs):
            if self.shuffle:
                self.shuffle_data()

            producer = ProducerThread(self.words, self.context, self.labels, self.batch_size, self.value, self.q, pct)
            # # print ("Starting producer")
            producer.start()
            # t = time.time()
            # while not self.q.full():
            #     pass 
            # print (time.time() - t)
            # exit()
            
  
            max_ = int(len(self.words) * pct)
            # print(max_)
            min_ = 0
            # if e + 1 % 2:
            #     max_ = int(len(self.words) * .50)
            #     min_ = int(len(self.words) * .25)
            # elif e + 1 % 3:
            #     max_ = int(len(self.words) * .75)
            #     min_ = int(len(self.words) * .50)
            # elif e + 1 % 4:
            #     max_ = int(len(self.words) * 1)
            #     min_ = int(len(self.words) * .75)
                
            self.experiment.set_step((z) * epochs + e + 1)

            

            # bar = progressbar.ProgressBar()
            loss = 0
            accuracy = 0
            c = 0
            bar = ""
            
            length = 0
            first = True 
            start = time.time()

            print ("\n")
            for i in range(min_, max_, self.batch_size):
                if c:
                    time_per_unit = (time.time() - start) / c 
                else:
                    time_per_unit = 0
                eta = time_per_unit * (max2 - c)
                a = time.time()

                x = self.q.get()
                x1 = x[0]
                x2 = x[1]
                y = x[2]

                # x1 = self.words[i:i+self.batch_size]
                # x2 = self.context[i:i+self.batch_size]
                # y = self.labels[i:i+self.batch_size]

                a_res = time.time() - a
                b = time.time()

                out = self.model.train_on_batch([x1, x2], y)
                b_res = time.time() - b

                # print ("\n")
                # print ("\n")
                # print (a_res)
                # print ("\n")
                # print (b_res)
                # exit()
                loss += out[0]
                accuracy += out[1]
                c += 1

                l = str(loss/c)[:7]
                a = str(accuracy/c)[:7]

                
                if c % mod == 0:
                    length += 1
                # print (length)

                bar = "|" + "#" * length + " " * (int(max2/mod) - length) + "|"

                sys.stdout.write("\reta: %s epoch %s:%s/%s %s loss: %s | acc: %s | loaded: %s" % (self.format_eta(eta), z, (e + 1), epochs, bar, l, a, loaded))
                sys.stdout.flush()
                # print ("\n")
                # self.K.clear_session()
                if l == 0 or a == 1:
                    
                    metrics = {
                        "loss": l,
                        "accuracy": a
                    }

                    self.experiment.log_metrics(metrics)
                    # self.model.save("trained/model_%s.h5" % e)

            
            metrics = {
                "loss": l,
                "accuracy": a
            }
            # print (time.time() - start)
            self.experiment.log_metrics(metrics)
            self.model.save("trained/model_%s.h5" % e)
            self.model.save("model2.h5")
            # keras.models.save_model(self.model, "model.h5")
            # self.model.save_weights('model_weights.h5')

            # # Save the model architecture
            # with open('model_architecture.json', 'w') as f:
            #     f.write(self.model.to_json())
            # self.word2vec.save("w2v.h5")
            
            # if e == 2:
            #     exit()


            # print (names)
            # for name, weight in zip(names, weights):
            #     print (name)
            #     if name=="embedding":
            #         print(2, name, weight)
            # exit()
            # print ("" % ())
            # exit()

        