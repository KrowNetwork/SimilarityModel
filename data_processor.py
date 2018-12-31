import json
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unicodedata
import math

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


data = []
with open('data.json') as f:
    for line in f:
        data.append(json.loads(line))

documents = []

for i in data:
    desc = i["description"].lower()
    documents.append(desc)

letters = {
    "a": 00,
    "b": '01',
    "c": '02',
    "d": '03',
    "e": '04',
    "f": '05',
    "g": '06',
    "h": '07',
    "i": '08',
    "j": '09',
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "q": 16,
    "r": 17,
    "s": 18,
    "t": 19,
    "u": 20,
    "v": 21,
    "w": 22,
    "x": 23,
    "y": 24,
    "z": 25,
    "UNK": 26,
    "PAD": 27
}

def clean(docs):
    docs2 = []
    vocab = []
    for i in docs:
        words2 = []
        tokens = word_tokenize(i)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        for w in words:
            if w not in stop_words:
                w = unicodedata.normalize('NFD', w)
                w = u"".join([c for c in w if not unicodedata.combining(c)])
                words2.append(w)
        words = words2
        # words = [porter.stem(word) for word in words]
        vocab.extend(words)
        docs2.append(" ".join(words))

    return docs2, list(set(vocab))
def subword_embedding(string):
    # w2s = {}
    # s2c = {}

    s = "<" + string + ">"
    arr = []
    for a in range(len(s) - 2):
        arr.append(s[a:a+3])
    # for a in range(len(a)):

    return arr
    

def create_conversion(vocab):
    w2v = {}
    v2w = {}

    for word in vocab:
        str_int = '1'
        for w in word:
            v = letters[w]
            str_int += str(v) 
        v2w[str_int] = word 
        w2v[word] = str_int
    return w2v, v2w

def subword(x, w2v):
    w = []
    # print (x)
    for a in subword_embedding(x):
        try:
            w.append(w2v[a])
        except:
            w.append(len(w2v))

    return w

def create_dataset(documents, w2v, n, context_type="before"):
    words, context = [], []
    for i in documents:
        split_doc = i.split(" ")
        c_ = []
        for z in range(len(split_doc) - n):

            if context_type == "middle":
                words.append(split_doc[z+math.floor(n/2)])
                c_ = []
                a = split_doc[z:z+math.floor(n/2)]
                a.extend(split_doc[z+math.ceil(n/2):z+n+1])
                for i in a:
                    c_.append(i)
                # context.append(c_)
                # print (words[0])
                # print (context[0])
                # exit()
            else:

                words.append(subword(split_doc[z+n], w2v))
                c_ = []
                for i in split_doc[z:z+n]:
                    c_.append(subword(i, w2v))
                context.append(c_)

    return words, context

def run():
    print ("cleaning documents and creating vocab")

    cleaned_docs, vocab = clean(documents)
    w2v = []
    v2w = {}
    w2v2 = {}
    for a in vocab:
        v = subword_embedding(a)
        for i in v:
            w2v.append(i)
    w2v = list(set(w2v))
    # print (w2v)
    for a in range(len(w2v)):
        v2w[a] = w2v[a]
        w2v2[w2v[a]] = a
    print (w2v2)
    l = len(w2v2)


    w2v2["PAD"] = l
    print(w2v2)
    # print (v2w)
    # print (w2v2)
    # exit()
        # w2v[a] = subword_embedding(a)

    # w2v, v2w = create_conversion(vocab)
    # print (w2v)
    # print (len(vocab))
    #                         # w2n, n2w = create_dictionary(vocab)
    # pickle.dump(vocab, open("vocab.bin", "wb"))
    pickle.dump(cleaned_docs, open("clean.bin", "wb"))
    pickle.dump(w2v2, open("w2v.bin", "wb"))
    pickle.dump(v2w, open("v2w.bin", "wb"))

# pickle.dump(vecs, open("vec/s.bin", "wb"))

# run()
# pickle.dump(n2w, open("n2w.bin", "wb"))
# print (vocab[:100])
# print (len(vocab))

