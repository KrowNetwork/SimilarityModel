import data_processor
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from scipy.spatial.distance import cosine

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import pprint
class W2VDocument():

    @staticmethod
    def dist(point1, point2):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point1[0], point1[1]

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
        return distance

    def __init__(self, document, model, w2v):
        self.document = document
        self.model = model
        self.w2v = w2v
        self.cleaned_doc = data_processor.clean([self.document])[0]
        self.split_doc = [d.split(" ") for d in self.cleaned_doc]
        self.subwords = [[data_processor.subword(w, w2v) for w in d] for d in self.split_doc]
        self.padded = [pad_sequences(d, maxlen=25, value=len(w2v), padding="post") for d in self.subwords]
        
        self.results = self.model.predict(self.padded)
        self.chunks = False
        self.score = 0
        # self.results = self.model.predict(self.padded)[0]
        # print (self.results)

    def __str__(self):
        return " ".join(self.split_doc[0][0:10])[:20] + " | Score: " + str(self.score)[:8]

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.document == other.document

    def _test(self, x, y):
        x = data_processor.subword(x, self.w2v)
        # print (x)
        y = data_processor.subword(y, self.w2v)

        x = pad_sequences([x], maxlen=25, value=len(self.w2v), padding="post")
        y = pad_sequences([y], maxlen=25, value=len(self.w2v), padding="post")

        wx = self.model.predict(x)
        wy = self.model.predict(y)

        # print (wx.shape)

        # print (wx)

        print (1 - cosine(wx, wy))

    def sum(self, arr = None):
        # print (self.results)
        if type(arr) == type(None):
            arr = self.results
        arr /= np.linalg.norm(arr)
        x = sum(arr)#/len(arr)
        
        return x

    

    def chunk_source_n(self, n):
        chunks = []
        for i in range(0, len(self.results) - n):
            x = self.results[i:i+n]
            chunks.append(self.sum(x))

        self.results = chunks
        self.chunks = True


        
        

    def calculate_similarity(self, w2v_):
        if not self.chunks:
            v1 = self.sum()
            v2 = w2v_.sum()
            sim = 1 - cosine(self.sum(), w2v_.sum())
            # print (sim)
            # sim = (0.25 * sim) + (1.25 * (sim ** 2))
            # sim = 1.0288 + (0.0002860917 - 1.0288)/(1 + (sim/0.7662593)**14.08159)
            if sim > 1:
                sim = 0.99
            if sim < 0:
                sim = 0.01
            # sim = (sim - -1)/ (1 - -1)
            # print (type(sim))
            if sim == np.nan:
                sim = 0
            return sim
        else:
            v2 = w2v_.sum()
            v2 = [v2]
            sims = []

            for i in self.results:
                for a in v2:
                    y = 1 - cosine(i, a)
                    if y > 1:
                        y = 1
                    elif y < -1:
                        y = 0
                    sims.append(y)
            # print (sims)
            # print (max(sims))
            return sum(sims)/len(sims)

    def display_pca(self, w2vs):
        points = []
        text = []
        vecs = []
        distances = []
        w2vs.append(self)

        pca = PCA(n_components=2)

        fig, ax = plt.subplots()


        for i in w2vs:
            s = i.sum()
            vecs.append(s)
            text.append(str(i))

        vecs = pca.fit_transform(vecs)
        for i, w in zip(vecs, text):
            points.append([i,w])

        for i in range(len(points) - 1):
            distances.append([points[i][0], points[i + 1][0]])

        for i in points:
            ax.scatter(i[0][0], i[0][1])
            ax.annotate(i[-1], (i[0][0], i[0][1]))

        plt.show()
        w2vs.pop(-1)
    def display_w2v_tsne(self):
        points = []
        text = []
        vecs = []
        distances = []

        tsne = TSNE(n_components=2, random_state=0, method="exact", init="pca")

        fig, ax = plt.subplots()

        for i in self.split_doc:
            text.extend(i)

        res = []
        for i in self.results:
            res.append(i/np.linalg.norm(i))

        vecs = tsne.fit_transform(res)
        for i, w in zip(vecs, text):
            points.append([i,w])

        for i in points:
            ax.scatter(i[0][0], i[0][1])
            ax.annotate(i[-1], (i[0][0], i[0][1]))

        plt.show()
    
    def create_matrix(self, w2vs):

        w2vs.append(self)

        vecs = []
        for i in w2vs:
            vecs.append(i.sum())

        # vecs = vecs[:7]

        cosine_sim = np.zeros((len(vecs), len(vecs)))

        for a in range(len(vecs)):
            for b in range(a, len(vecs)):
                # print (1 - cosine(vecs[a], vecs[b]))
                cosine_sim[a][b] = 1 - cosine(vecs[a], vecs[b])
                cosine_sim[b][a] = 1 - cosine(vecs[a], vecs[b])

        # print (cosine_sim)
        matrix = cosine_sim
        s = [[str(e * 1)[:4] for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print ('\n'.join(table))
        w2vs.pop(-1)

        for i, a in zip(matrix[-1], w2vs):
            a.score = i

        w2vs.pop(-1)
        # return matrix[-1]




        
    

