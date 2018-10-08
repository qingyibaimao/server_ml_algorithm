import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

x, y_true = make_blobs(centers= 4, n_samples= 5000)
# fig = plt.figure(figsize= (8, 6))
# plt.scatter(x[:, 0], x[:, 1], c= y_true)
# plt.title("dataset")
# plt.xlabel("first feature")
# plt.ylabel("second feature")
# plt.show()

y_true = y_true[:, np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(x, y_true)

class SoftmaxRegressor:

	def __init__(self):
		pass

	def train(self, x, y_true, n_classes, n_iters= 10, learning_rate= 0.1):	
		self.n_samples, n_features = x.shape
		self.n_classes = n_classes
		self.weights = np.random.rand(self.n_classes, n_features)
		self.bias = np.zeros((1, self.n_classes))
		all_losses = []
		for i in range(n_iters):
			# 计算值
			scores = self.compute_scores(x)
			# 用softmax 将值转化为概率值
			probs = self.softmax(scores)
			y_predict = np.argmax(probs, axis= 1)[:, np.newaxis]
			y_one_hot = self.one_hot(y_true)
			# 求损失
			loss = self.cross_entropy(y_one_hot, probs)
			all_losses.append(loss)
			# 进行梯度运算
			dw = (1 / self.n_samples) * np.dot(x.T, (probs - y_one_hot))
			db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis= 0)
			# 更新w,b
			self.weights = self.weights - learning_rate * dw.T
			self.bias = self.bias - learning_rate * db
			if i % 100 == 0:
				print(f'iteration number : {i}, loss: {np.round(loss, 4)}')
		return self.weights, self.bias, all_losses

	def predict(self, x):
		scores = self.compute_scores(x)
		probs = self.softmax(scores)
		return np.argmax(probs, axis= 1)[: , np.newaxis]

	def softmax(self, scores):
		# 将预测得分转化为概率
		exp = np.exp(scores)
		sum_exp = np.sum(np.exp(scores), axis= 1, keepdims= True)
		softmax = exp / sum_exp
		return softmax

	def compute_scores(self, x):
		return np.dot(x, self.weights.T) + self.bias

	def cross_entropy(self, y_true, scores):
		loss = - (1 / self.n_samples) * np.sum(y_true * np.log(scores))
		return loss

	def one_hot(self, y):
		one_hot = np.zeros((self.n_samples, self.n_classes))
		one_hot[np.arange(self.n_samples), y.T] = 1
		return one_hot

regressor = SoftmaxRegressor()
w_trained, b_trained, loss = regressor.train(x_train, y_train, learning_rate= 0.1, n_iters= 800, n_classes= 4)
# fig = plt.figure(figsize= (8, 6))
# plt.plot(np.arange(800), loss)
# plt.title("development of loss during training")
# plt.xlabel("number of iterations")
# plt.ylabel("loss")
# plt.show()

n_test_samples, _ = x_test.shape
y_predict = regressor.predict(x_test)
print(f"classification accuracy on test set: {(np.sum(y_predict == y_test) / n_test_samples) * 100}%")