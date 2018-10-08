import numpy as np 
import matplotlib.pyplot as plt
import random 
from sklearn.datasets import make_blobs

#data
x, y = make_blobs(centers= 4, n_samples= 1000)
print(f'shape of dataset: {x.shape}')

#绘图
# fig = plt.figure(figsize= (8, 6))
# plt.scatter(x[:, 0], x[:, 1], c= y)
# plt.title("dataset with 4 clusters")
# plt.xlabel("first feature")
# plt.ylabel("second feature")
# plt.show()

#k_means:
class KMeans():

	def __init__(self, n= 4):
		self.k = n

	def fit(self, data):
		n_samples, _ = data.shape
		self.centers = np.array(random.sample(list(data), self.k))# 用于从总体序列中选取k长度的唯一元素列表(随机初始化4个点)
		#print(self.centers)
		self.initial_centers = np.copy(self.centers)  # 记下这四个点

		old_assigns = None
		n_iters = 0

		while True:
			new_assign = [self.classify(datapoint) for datapoint in data]  # 计算每个点离那个分配点最近
			#print(new_assign)
			if new_assign == old_assigns:
				print(f"Training finished after {n_iters} iterations")
				return

			old_assigns = new_assign
			n_iters += 1

			for id in range(self.k):  # 划分范围 
				points_idx = np.where(np.array(new_assign) == id)
				datapoints = data[points_idx]
				self.centers[id] = datapoints.mean(axis= 0)  # mean取均值 axis=0 压缩行, axis=1 压缩列,对各行求均值返回 m*1阵


	def distance(self, datapoint):  # 计算距离
		dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis= 1))
		return dists


	def classify(self, datapoint):

		dists = self.distance(datapoint)
		return np.argmin(dists)

	def plot_clusters(self, data):
		plt.figure(figsize= (8,6))
		plt.title("initial centers in black, final centers in red")
		plt.scatter(data[:, 0], data[:, 1], marker= '.', c= y)
		plt.scatter(self.centers[:, 0], self.centers[:, 1], c= 'r')
		plt.scatter(self.initial_centers[:, 0], self.initial_centers[:, 1], c= 'k')
		plt.show()

kmeans = KMeans(n= 4)
kmeans.fit(x)
kmeans.plot_clusters(x)

































