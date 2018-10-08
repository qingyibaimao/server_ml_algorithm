import numpy as np 
import matplotlib.pyplot as plt
import random 
from sklearn.datasets import make_blobs

#data
x, y = make_blobs(centers= 4, n_samples= 1000)
print(f'shape of dataset: {x.shape}')
print(f'shape of dataset: {y.shape}')

#绘图
fig = plt.figure(figsize= (8, 6))
plt.scatter(x[:, 0], x[:, 1], c= y)
plt.title("dataset with 4 clusters")
plt.xlabel("first feature")
plt.ylabel("second feature")
plt.show()