import random
import numpy as np 
import keras
import progressbar
import sys
import time
from model import sum_embedding

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

    def __init__(self, K, model, word2vec, words, context, labels, value, experiment, batch_size=2048, shuffle=False):
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

    def on_epoch_end(self):
        if self.shuffle == True:
            self.words, self.context, self.labels = self.randomize(self.words, self.context, self.labels)

    def shuffle_data(self):
        self.words, self.context, self.labels = self.randomize(self.words, self.context, self.labels)

    def train(self, epochs):
        mod = int(np.floor(np.log10(len(self.words) / self.batch_size))) + 1
        self.words = self.pad(self.words, self.value)
        self.context = self.pad(self.context, self.value)

        max_ = max(range(0, len(self.words), self.batch_size))
        max2 = len(range(0, len(self.words), self.batch_size))
        width = max2 % mod
        begin = time.time()
        for e in range(epochs):
            self.experiment.set_step(e + 1)

            if self.shuffle:
                self.shuffle_data()

            # bar = progressbar.ProgressBar()
            loss = 0
            accuracy = 0
            c = 0
            bar = ""
            
            length = 0
            first = True 
            start = time.time()
            print ("\n")
            for i in range(0, len(self.words), self.batch_size):
                if c:
                    time_per_unit = (time.time() - start) / c 
                else:
                    time_per_unit = 0
                eta = time_per_unit * (max2 - c)

                x1 = self.words[i:i+self.batch_size]
                x2 = self.context[i:i+self.batch_size]
                y = self.labels[i:i+self.batch_size]

                out = self.model.train_on_batch([x1, x2], y)
                loss += out[0]
                accuracy += out[1]
                c += 1

                l = str(loss/c)[:7]
                a = str(accuracy/c)[:7]

                
                if c % mod == 0:
                    length += 1
                # print (length)

                bar = "|" + "#" * length + " " * (int(max2/mod) - length) + "|"

                sys.stdout.write("\reta: %s epoch %s/%s %s loss: %s | acc: %s" % (self.format_eta(eta), (e + 1), epochs, bar, l, a))
                sys.stdout.flush()
                # self.K.clear_session()
                if l == 0 or a == 1:
                    
                    metrics = {
                        "loss": l,
                        "accuracy": a
                    }

                    self.experiment.log_metrics(metrics)
                    self.model.save("model.h5")

            

            metrics = {
                "loss": l,
                "accuracy": a
            }
            self.experiment.log_metrics(metrics)
            # self.model.save("model.h5")
            keras.models.save_model(model, "model.h5")
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

        