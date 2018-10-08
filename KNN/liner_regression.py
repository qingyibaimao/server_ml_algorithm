import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# np.random.seed(123)
x = 2* np.random.rand(500, 1)
y = 5+ 3* x + np.random.randn(500, 1)
#fig = plt.figure(figsize= (8, 6))
# plt.scatter(x, y)
# plt.title("dataset")
# plt.xlabel("first feature")
# plt.ylabel("swcond feature")
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y)
# print(f'ahape x_train: {x_train.shape}')
# print(f'shape y_train: {y_train.shape}')
# print(f'shape x_test: {x_test.shape}')
# print(f'shape y_test: {y_test.shape}')

class LinearRegression:     #线性回归
	
	def __init__(self):
		pass
		
	# 梯度下降
	def train_gradient_descent(self, x, y, learning_rate= 0.01, n_iters= 100):  
		# 初始化w,b
		n_samples, n_features = x.shape
		self.weights = np.zeros(shape= (n_features, 1))  
		self.bias = 0
		costs = []

		for i in range(n_iters):
			# y = wx+b
			y_predict = np.dot(x, self.weights) + self.bias   
			# 损失函数
			cost = (1 / n_samples) * np.sum((y_predict- y)**2)
			costs.append(cost)

			if i % 100 == 0:
				print(f"cost at iteration {i} : {costs}\n\d")
			# 求偏导分
			dj_dw = (2 / n_samples) * np.dot(x.T, (y_predict - y))
			dj_db = (2 / n_samples) * np.sum((y_predict - y))
			#更新w.b
			self.weights = self.weights - learning_rate * dj_dw
			self.bias = self.bias - learning_rate * dj_db

		return self.weights, self.bias, costs
	# 正规方程
	def train_normal_equation(self, x, y):
		self.weights = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y) #(xT*x)(-1)*xT*y
		self.bias = 0
		return self.weights, self.bias

	def predict(self, x):
		return np.dot(x, self.weights) + self.bias



#梯度画图
# regressor = LinearRegression()
# w_trained, b_trained, costs = regressor.train_gradient_descent(x_train, y_train, learning_rate= 0.005, n_iters= 600)
# fig = plt.figure(figsize =(8, 6))
# plt.plot(np.arange(600), costs)
# plt.title("development of cost druing training")
# plt.xlabel("number of iteration")
# plt.ylabel("cost")
# plt.show()

#梯度测试

# n_samples, _ = x_train.shape
# n_samples_test, _ = x_test.shape

# y_p_train = regressor.predict(x_train)
# y_p_test = regressor.predict(x_test)

# error_train = (1 / n_samples) * np.sum((y_p_train - y_train)**2)
# error_test = (1 / n_samples_test) * np.sum((y_p_test - y_test)**2)

# print(f"error on training set: {np.round(error_train, 4)}")
# print(f"error on test set: {np.round(error_test)}")


#正规方程训练
n_samples, _ = x_train.shape
n_samples_test, _ = x_test.shape

x_b_train = np.c_[np.ones((n_samples)), x_train]
x_b_test = np.c_[np.ones((n_samples_test)), x_test]

reg_normal = LinearRegression()
w_trained = reg_normal.train_normal_equation(x_b_train, y_train)

#正规方程测试

y_p_train = reg_normal.predict(x_b_train)
y_p_test = reg_normal.predict(x_b_test)

# error_train = (1 / n_samples) * np.sum((y_p_train - y_train)**2)
# error_test = (1 / n_samples_test) * np.sum((y_p_test - y_test)**2)

# print(f"error on training set: {np.round(error_train, 4)}")
# print(f"error on test set: {np.round(error_test)}")

#正规方程画图

fig = plt.figure(figsize= (8, 6))
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_p_test)
plt.xlabel("first feature")
plt.ylabel("second feature")
plt.show()