import tensorflow as tf 
import numpy as np
# import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(Model, self).__init__()

        self.vocab = tf.contrib.eager.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[vocab_size+1, embedding_size]), trainable=True)
        self.dense = tf.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dot = tf.keras.layers.Dot(1)

    def predict(self, words, context):
        emb_words = tf.nn.embedding_lookup(self.vocab, words)
        # print (emb_words)
        emb_context = tf.nn.embedding_lookup(self.vocab, context)
        # print (emb_context)

        context = tf.reduce_sum(tf.reduce_sum(emb_context, axis=1), axis=1)
        # print (context)
        words = tf.reduce_sum(emb_words, axis=1)
        # print (words)
        dot_prod = self.dot([context, words])
        dot_prod = self.dense(dot_prod)
        return dot_prod

    def loss_fn(self, words, context, labels):
        logits = self.predict(words, context)
        labels = np.reshape(labels, (len(labels), 1))
        loss = tf.reduce_sum(tf.square(tf.subtract(labels, logits)))
        print (loss)
        # labels = labels.reshape(len(labels),)
        # logits = tf.reshape(logits, (len(labels),))
        # print (labels)
        # logits = tf.cast(logits, tf.float32)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss = tf.keras.backend.binary_crossentropy(labels, logits)
        equality = tf.equal(logits, labels)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        # print (accuracy)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return loss, accuracy

    def grads_fn(self, words, context, labels):
        with tf.GradientTape() as tape:
            loss, accuracy = self.loss_fn(words, context, labels)
            # print (self.variables)
        # print (len(self.trainable_variables))
        c = tape.gradient(loss, self.trainable_variables)
        # print (c)
        return c, loss, accuracy

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_on_batch(self, words, context, labels):
        # print (tf.all_variables())
        # variables_names = [v for v in tf.trainable_variables()]
        # print (len(variables_names))
        # exit()
        grads, loss, accuracy = self.grads_fn(words, context, labels)
        self.optimizer.apply_gradients(zip(grads, self.variables))
        # print (len(tf.trainable_variables()))
        return loss, accuracy


# m = Model(128, 10)
# target = 1
# print (m.predict([[10]], [[120, 110]]))
# print (m.loss_fn([[10]], [[120, 110]], [[1]]))