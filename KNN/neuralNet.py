import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

x, y = make_circles(n_samples= 1000, factor= 0.5, noise= .1)
# fig = plt.figure(figsize= (8, 6))
# plt.scatter(x[:, 0], x[:, 1], c= y)
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
# plt.title("datasets")
# plt.xlabel("first feature")
# plt.ylabel("second feature")
# plt.show()
y_true = y[:, np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(x, y_true)

class NeuralNet():

	def __init__(self, n_inputs, n_outputs, n_hidden):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.hidden = n_hidden

		self.w_h = np.random.randn(self.n_inputs, self.hidden)
		self.b_h = np.zeros((1, self.hidden))
		self.w_o = np.random.randn(self.hidden, self.n_outputs)
		self.b_o = np.zeros((1, self.n_outputs))

	def sigmoid(self, a):
		#输出层函数使用sigmoid函数	
		return 1 / (1 + np.exp(-a))

	def forward_pass(self, x):
		"""
			通过前向传播
			a_h : 隐藏神经元的激活函数
			o_h : 隐藏神经元的输出
			a_o : 输出神经元的激活函数
			o_o : 输出神经元的输出

		"""
		#计算隐藏单元的激活函数和输出
		a_h = np.dot(x, self.w_h) + self.b_h
		o_h = np.tanh(a_h)
		#计算输出单元的激活函数和输出
		a_o = np.dot(o_h, self.w_o) + self.b_o
		o_o = self.sigmoid(a_o)
		outputs = {
				"a_h": a_h,
				"a_o": a_o,
				"o_h": o_h,
				"o_o": o_o,
		}
		return outputs

	def cost(self, y_true, y_predict, n_samples):
		cost = (-1 / n_samples) * np.sum(y_true * np.log(y_predict) + (1 - y_true) * (np.log(1 - y_predict)))
		cost = np.squeeze(cost)
		assert isinstance(cost, float)
		return cost

	def backward_pass(self, x, y, n_samples, outputs):
		"""
			通过反向传播找到误差
			dw_h: 损失函数的偏导数和隐藏权重
			db_h: 损失函数的偏导数的隐藏偏置
			dw_o: 损失函数的偏导数的输出权重
			db_o: 损失函数的偏导数的输出偏置
		"""
		#对于输出层的梯度
		da_o = (outputs["o_o"] - y)
		dw_o = (1 / n_samples) * np.dot(outputs["o_h"].T, da_o)
		db_o = (1 / n_samples) * np.sum(da_o)
		#对于隐藏层的梯度
		da_h = (np.dot(da_o, self.w_o.T)) * (1 - np.power(outputs["o_h"], 2))
		dw_h = (1 / n_samples) * np.dot(x.T, da_h)
		db_h = (1 / n_samples) * np.sum(da_h)
		gradients= {
				"dw_o": dw_o,
				"db_o": db_o,
				"dw_h": dw_h,
				"db_h": db_h,
		}
		return gradients

	def update_weights(self, gradients, eta):
		#使用固定学习速率更新模型参数
		self.w_o = self.w_o - eta * gradients["dw_o"]
		self.w_h = self.w_h - eta * gradients["dw_h"]
		self.b_o = self.b_o - eta * gradients["db_o"]
		self.b_h = self.b_h - eta * gradients["db_h"]

	def train(self, x, y, n_iters= 500, eta= 0.3):
		n_samples, _ = x.shape
		for i in range(n_iters):
			outputs = self.forward_pass(x)
			cost = self.cost(y, outputs["o_o"], n_samples= n_samples)
			gradients = self.backward_pass(x, y, n_samples, outputs)
			if i % 100 == 0: 
				print(f"cost at iteration {i} : {np.round(cost, 4)}")
			self.update_weights(gradients, eta)

	def predict(self, x):
		# 计算给定数据集的网络预测
		outputs = self.forward_pass(x)
		y_pred = [1 if elem >= 0.5 else 0 for elem in outputs["o_o"]]
		return np.array(y_pred)[:, np.newaxis]

nn = NeuralNet(n_inputs= 2, n_hidden= 6, n_outputs= 1)
# 训练神经网络
# print("shape of weight matrices and bias vectors:")
# print(f"w_h shape: {nn.w_h.shape}")
# print(f'b_h shape: {nn.b_h.shape}')
# print(f"w_o shape: {nn.w_o.shape}")
# print(f"b_o shape: {nn.b_o.shape}")
# print()
# print("training:")
nn.train(x_train, y_train, n_iters= 2000, eta= 0.7)

#测试神经网络
n_test_samples = x_test.shape
y_predict = nn.predict(x_test)
print(f"classification accuracy on test set: {(np.sum(y_predict == y_test) / n_test_samples) * 100}%")

# 可视化决策边界

x_temp, y_temp = make_circles(n_samples= 60000, noise= .5)
y_predict_temp = nn.predict(x_temp)
y_predict_temp = np.ravel(y_predict_temp)

fig = plt.figure(figsize= (8, 10))
ax = fig.add_subplot(2, 2, 1)
plt.scatter(x[:, 0], x[:, 1], c= y)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel("first feature")
plt.ylabel("second feature")
ax = fig.add_subplot(2, 2, 2)
plt.scatter(x_temp[:, 0], x_temp[:, 1], c= y_predict_temp)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel("first feature")
plt.ylabel("second feature")
plt.title("decision boundary")
plt.show()

































