import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

#data
digits = load_digits()
x, y = digits.data, digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y)
print(f'X_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')

#示例数字
fig = plt.figure(figsize= (10, 8))
for i in range(10):
	ax = fig.add_subplot(2, 5, i+1)
	plt.imshow(x[i].reshape((8, 8)), cmap= 'gray')
plt.show()

#KNN
class KNN():

	def __init__(self):
		pass

	def fit(self, x, y):
		self.data = x
		self.targets = y

	def euclidean_distance(self, x):
		if x.ndim == 1:    # x.ndim返回数维度
			ld = np.sqrt(np.sum((self.data - x)**2, axis= 1))   # axis指维度,0为行, 1为列,通常在函数中指定axis=n时,
			#函数输出的数组中n维会被消去

		if x.ndim == 2:
			n_samples, _ = x.shape
			ld = [np.sqrt(np.sum((self.data - x[i])**2, axis= 1)) for i in range(n_samples)]

		return np.array(ld)

	def predict(self, x, k= 1):
		dists = self.euclidean_distance(x)

		if x.ndim == 1:
			if k == 1:
				nn = np.argmin(dists)  #返回最大值索引
				return self.targets[nn]
			else:
				knn = np.argsort(dists)[:k]
				y_knn = self.targets[knn]
				max_vote = max(y_knn, key= list(y_knn).count)
				return max_vote
		if x.ndim == 2:
			knn = np.argsort(dists)[:, :k]
			y_knn = self.targets[knn]
			if k == 1:
				return y_knn.T
			else:
				n_samples, _ = x.shape
				max_votes = [max(y_knn[i], key= list(y_knn[i]).count) for i in range(n_samples)]
				return max_votes

knn = KNN()
knn.fit(x_train, y_train)
# print("Testing one datapoint, k=1")
# print(f"Predicted label: {knn.predict(x_test[0], k=1)}")
# print(f"True label: {y_test[0]}")
# print()
# print("Testing one datapoint, k=5")
# print(f"Predicted label: {knn.predict(x_test[20], k=5)}")
# print(f"True label: {y_test[20]}")
# print()
# print("Testing 10 datapoint, k=1")
# print(f"Predicted labels: {knn.predict(x_test[5:15], k=1)}")
# print(f"True labels: {y_test[5:15]}")
# print()
# print("Testing 10 datapoint, k=4")
# print(f"Predicted labels: {knn.predict(x_test[5:15], k=4)}")
# print(f"True labels: {y_test[5:15]}")
# print()

#测试
y_p_test1 = knn.predict(x_test, k=1)
test_acc1 = np.sum(y_p_test1[0] == y_test)/ len(y_p_test1[0]) * 100
print(f"test accuracy with k= 1:{format(test_acc1)}")

y_p_test5 = knn.predict(x_test, k= 5)
test_acc5 = np.sum(y_p_test5 == y_test) / len(y_p_test5) * 100
print(f"test accuracy with k= 8 : {format(test_acc5)}")












