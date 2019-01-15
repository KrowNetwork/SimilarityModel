import numpy as np
x = (4096, 3, 25, 300)
arr = [[[[11, 15, 10], [11, 43, 12]]]]
arr = np.array(arr)
print(arr.shape)
print (arr)
a = np.tile(arr, [3, 1, 1])
print (a)
a = np.sum(a, axis=2)
print (a)
a = np.mean(a, axis=1)
print (a)