import random
import numpy as np
length = 8000000
size = 5000

success = []


# print (range_s)
round_ = 0
total = []
for e in range(5):
    range_ = np.array(list(range(8000000)))
    # print (range_.shape)
    range_s = list(range(8000000))
    print (e)
    for i in range(1000):
        print (i)
        x = list(random.sample(range_.tolist(), size))
        # x = range_[:size]
        for a in x:
            if a < len(range_) + 1:
                range_[a] = len(range_) + 1
                # print(len(range_))
        # range_ = x   
        # exit()

    rang_ = len(np.where(range_ == len(range_) + 1)[0])
    range_s = len(range_s)
    print (range_s)
    total.append(rang_/range_s)
print (total)
print (sum(total)/len(total))
    