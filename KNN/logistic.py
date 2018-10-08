import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# % matplotlib inline # 魔法函数, 可以内嵌视图,可以省略掉np.show()这一步

# dataset
x, y_true = make_blobs(n_samples= 1000, centers=2)
#绘图

# fig = plt.figure(figsize=(8,6))
# plt.scatter(x[:,0], x[:,1], c= y_true)
# plt.title("Dataset")
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# plt.show()
y_true = y_true[:, np.newaxis]
x_train, x_test, y_train, y_test= train_test_split(x, y_true)

class LogisticRegression:

	def __init__(self):
		pass

	def sigmoid(self, a):  #sigmoid函数
		return 1/ (1 + np.exp(-a))

	def train(self, x, y_true, n_iters, learning_rate):
		#初始化函数
		n_samples, n_features = x.shape
		self.weights = np.zeros((n_features, 1))
		self.bias = 0
		costs = []

		for i in range(n_iters):
			#主要函数
			y_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)
			#损失函数
			cost = (-1 / n_samples) * np.sum(y_true * np.log(y_predict) + (1 - y_true) * (np.log(1 - y_predict)))

			#求w,b
			dw = (1 / n_samples) * np.dot(x.T, (y_predict - y_true))
			db = (1 / n_samples) * np.sum(y_predict - y_true)

			#update w.b
			self.weights = self.weights - learning_rate * dw
			self.bias = self.bias - learning_rate * db

			costs.append(cost)
			if i % 100 == 0:
				print(f'cost agter iteration {i} : {cost}')


		return self.weights, self.bias, costs

	#预测二进制标签
	def predict(self, x):
		y_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)
		y_predict_labels = [1 if elem > 0.5 else 0 for elem in y_predict]

		return np.array(y_predict_labels)[:, np.newaxis]


# 初始化并训练模型
regressor = LogisticRegression()
w_trained, b_trained, costs = regressor.train(x_train, y_train, n_iters= 600, learning_rate=0.009)

# fig = plt.figure(figsize= (8, 6))
# plt.plot(np.arange(600), costs)
# plt.title("development of cost over training")
# plt.xlabel("number of iterations")
# plt.ylabel("cost")
# plt.show()	

#测试

# y_p_train = regressor.predict(x_train)
# y_p_test = regressor.predict(x_test)

# print(f"train accuracy: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
# print(f"test accuracy: {100 - np.mean(np.abs(y_p_test - y_test)) * 100}%")
















