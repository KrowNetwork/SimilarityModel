import threading
# import Queue
import time 
import random
import keras

class ProducerThread(threading.Thread):
    def __init__(self, words, context, labels, batch_size, vocab_size, q):
        super(ProducerThread,self).__init__()
        self.words = words
        self.context = context 
        self.labels = labels 
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.q = q
        
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

    def run(self):
        i = 0
        while i < len(self.words):
            if not self.q.full():
                x1 = self.words[i:i+self.batch_size]
                x2 = self.context[i:i+self.batch_size]
                y = self.labels[i:i+self.batch_size]
                i += self.batch_size
                t = time.time()
                self.q.put([x1, x2, y])
                # print (time.time() - t)
                # print ("added")
                # time.sleep(random.random())

        return