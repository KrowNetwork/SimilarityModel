from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
import multiprocessing as mp

class Generator(Sequence):
    def __init__(self, words, context, labels, batch_size, vocab_size):
        self.words = words
        self.context = context 
        self.labels = labels 
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    @staticmethod
    def pad(x, value):
        y = []
        if type(x[0][0]) != type(0):
            c_ = []
            for i in x:
                c_.append(list(pad_sequences(i, maxlen=25, value=value, padding="post")))    
            y = c_
        else:
            y = list(pad_sequences(x, maxlen=25, value=value, padding="post"))
        return y

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        q = mp.Queue(maxsize=10)
        def _gen_batches():
            num_samples = len(self.words)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batch_size + 1, self.batch_size)
            for batch in batches:
                batch_words = self.words[idx[batch:batch + self.batch_size]]
                batch_context = self.context[idx[batch:batch + self.batch_size]]
                batch_labels = self.labels[idx[batch:batch + self.batch_size]]
          
                # do some stuff to the batches like augment images or load from folders
                
                yield [batch_words, batch_context, batch_labels]

        def produce(gen):
            batch_gen = gen() 
            for i in batch_gen:
                q.put(data)

            
        

        batch_words = pad(batch_words, self.vocab_size)
        batch_context = pad(batch_words, self.vocab_size)

    

        

        