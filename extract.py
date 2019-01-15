from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.spatial.distance import cosine
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import data_processor
import pickle 
import numpy as np
import matplotlib.pyplot as plt
# from keras import backend as K
import tensorflow as tf 

def calculate_similarity(v1, v2):
    sim = 1 - cosine(v1, v2)
    # print (sim)
    # sim = (0.25 * sim) + (1.25 * (sim ** 2))
    # sim = 5.5511150000000004e-17 + 0.375*sim + 0.9375*sim**2
    # if sim > 1:
    #     sim = 0.99
    # if sim < 0:
    #     sim = 0.01
    # sim = (sim - -1)/ (1 - -1)
    # print (type(sim))
    if sim == np.nan:
        sim = 0
    return sim

K = tf.keras.backend
load_model = tf.keras.models.load_model

x = load_model("model.h5")
print ([layer.name for layer in x.layers])

# word2vec = load_model("w2v.h5")

word2vec = tf.keras.models.Model(inputs=x.input[0], outputs=x.get_layer("embedding").output)
# word2vec.save("test.h5")
# word2vec = load_model("test.h5")
# word2vec.compile(optimizer=keras.optimizers.Adam())
del x

resume = '''
Krow Network, LLC August 2018 to Present As Co-founder and CTO of Krow Network, I lead the development of our platform. Experienced in Angular (Javascript and HTML), Python, and other languages. I also manage our servers hosted through AWS and Google Cloud, and run all machine learning/artificial intelligence projects. Skills: Managment, Technology , Computer Science , Web Design , AI , Python , Angular 
Best Buy October 2018 to Present Serve on the sales floor, selling connected home devices, such as routers, smart home devices, audio equipment, and more. Responsible for building and maintaining customer relationships, and up selling services to benefit the customer. Skills: Sales, Technology , Customer Support , Amazon Alexa , Audio Products , Google Home , Networking 
Gold Medal Fitness May 2017 to Present Serve at the front desk of Gold Medal Fitness Garwood and Cranford. Responsibilities include managing member accounts, taking phone calls, managing member relations, and selling new memberships Skills: Sales, Member Managment , Fitness 
'''

resume = data_processor.clean(resume.lower().split("."))[0]
resume = [r for r in resume if r != ""]
w2v = pickle.load(open("w2v.bin", "rb"))
# print (resume)

sd = np.array(resume).tolist()
docs = [[data_processor.subword(w, w2v) for w in d] for d in resume]
print (docs)
## Create subwords

# docs = [data_processor.subword(w, w2v) for w in docs]

## Pad
for i in range(len(docs)):
    docs[i] = pad_sequences(docs[i], maxlen=25, value=len(w2v), padding="post")
# print (sd)
results = []

for i in docs:
    # print (i)
    x = word2vec.predict([i])
    for i in range(len(x)):
        a = x[i]
        a_d = np.linalg.norm(a)
        a = a / a_d
        x[i] = a

    results.append(sum(x)/len(x))

print (results[0])
print (len(results))
print (len(docs))

sims = []
for i in range(0, len(results) - 1):
    r1 = results[i]
    r2 = results[i + 1]

    sims.append(calculate_similarity(r1, r2))

for i, a in enumerate(sims):
    print (i, a)

print (sd[:8])
# print (sd[43:60])

n_sne = 7000
tsne = TSNE(n_components=2, perplexity=5, n_iter=30000)

t_results = tsne.fit_transform(results)

x = []
y = []
for value in t_results:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(sd[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
plt.show()