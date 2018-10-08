import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#data
x, y = make_blobs(n_samples= 1000, centers= 2)
y_true = y[:, np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(x, y_true)
#绘图
# fig = plt.figure(figsize= (8, 6))
# plt.scatter(x[:, 0], x[:, 1], c= y)
# plt.title("dataset")
# plt.xlabel("first feature")
# plt.ylabel("second feature")
# plt.show()

#感知器分类
class Perceptron():

	def __init__(self):
		pass

	def train(self, x, y, learning_rate= 0.05, n_iters= 100):
		n_samples, n_features = x.shape

		self.weights = np.zeros((n_features, 1))
		self.bias = 0

		for i in range(n_iters):
			#激活函数
			a = np.dot(x, self.weights) + self.bias
			#计算输出
			y_predict = self.step_function(a)

			delta_w = learning_rate * np.dot(x.T , (y - y_predict))
			delta_b = learning_rate * np.sum(y - y_predict)

			self.weights += delta_w
			self.bias += delta_b

		return self.weights, self.bias

	def step_function(self, x):
		return np.array([1 if elem >= 0 else 0 for elem in x])[:, np.newaxis]

	def predict(self, x):
		a = np.dot(x, self.weights) + self.bias
		return self.step_function(a)


#初始化和训练模型
p = Perceptron()
# for i in arange(0, 0.5, 0.01):
# 	w_trained, b_trained = p.train(x_train, y_train, i, n_iters= 500)
# 	y_p_train = p.predict(x_train)
# 	y_p_test = p.predict(x_test)

# 	print(f"learning_rate is {i} the traing accurary: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
# 	print(f"learning_rate is {i} the test accurary: {100 - np.mean(np.abs(y_p_test - y_test)) * 100}%")
w_trained, b_trained = p.train(x_train, y_train, 0.05, n_iters= 500)
y_p_train = p.predict(x_train)
y_p_test = p.predict(x_test)

print(f"the traing accurary: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
print(f"the test accurary: {100 - np.mean(np.abs(y_p_test - y_test)) * 100}%")

#可视化决策边界
def plot_hyperplane(x, y, weights, bias):
	slope = - weights[0] / weights[1]
	intercept = - bias / weights[1]
	x_hyperplane = np.linspace(-10, 10, 10)
	y_hyperplane = slope * x_hyperplane + intercept
	fig = plt.figure(figsize= (8, 6))
	plt.scatter(x[:, 0], x[:, 1], c= y)
	plt.plot(x_hyperplane, y_hyperplane, '-')
	plt.title("dataset adn fitted decision hyperplane")
	plt.xlabel("first feature")
	plt.ylabel("second feature")
	plt.show()

plot_hyperplane(x, y, w_trained, b_trained)