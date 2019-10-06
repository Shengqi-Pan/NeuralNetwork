# -*- coding: utf-8 -*-
#%%
import numpy as np
import math
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, lr = 0.01, epochs = 50):
        self.lr = lr
        self.epochs = epochs
        # 权重初始化
        self.wi = np.random.rand(10)  # 隐藏层的权值矩阵
        self.wi = np.array([-0.1, 0.13,-0.23, 0.18, -0.26, -0.13, 0.031, 0.02, 0.011, 0.021])
        self.bi = np.random.rand(10)      # 隐藏层阈值
        self.wo = np.random.rand(10)  # 输出层的权值矩阵
        self.bo = 0.3                       # 输出层阈值

    def train(self, x, y):
        # 迭代
        self.errors_ = []
        for i in range (self.epochs):
            # 初始化误差
            errors = 0
            # 训练
            for xi, target in zip(x, y):
                # 隐藏层输入
                neth1 = xi * self.wi + self.bi
                # 隐藏层输出
                outh1 = self.sigmoid(neth1)
                # 输出层输入
                neto1 = np.inner(outh1, self.wo) + self.bo
                # 模型输出
                outo1 = self.sigmoid(neto1)
                
                # 计算误差
                errors = 0.5 * (target - self.predict(xi)) ** 2
                # self.errors_.append(errors)

                # de/douth1
                d1 = - outo1 * (target - outo1) * self.rsigmoid(neto1) * self.wo
                
                # douth1/dx
                d2 = xi * self.rsigmoid(neth1)

                # 反向传播
                # 更新输出层权值
                self.wo = self.wo - self.lr * (-(target - outo1) * self.rsigmoid(neto1)) * outh1               
                self.bo = self.bo - self.lr * (-(target - outo1) * self.rsigmoid(neto1))
                # 更新隐藏层权值
                self.wi = self.wi - self.lr * np.multiply(d2, d1)
                self.bi = self.bi - self.lr * np.multiply(self.rsigmoid(neth1), d1)
            self.errors_.append(errors)
            print(errors)
        return self

    # 激活函数
    def sigmoid(self, x):
        return 2 / (1 + np.exp(-x))

    def newsigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 激活函数求导
    def rsigmoid(self, x):
        return 2 * np.multiply(self.newsigmoid(x), (1 - self.newsigmoid(x)))

    # 第一层的输出
    def layer1out(self, x):
        return self.sigmoid(x * self.wi + self.bi)

    # 预测
    def predict(self, x):
        return self.sigmoid(np.inner(self.layer1out(x), self.wo) + self.bo)

# 准备数据
# x = np.pi * np.random.rand(10)
x = np.arange(0, np.pi + np.pi/10, np.pi/10)
y = np.sin(x)
plt.plot(x, y)
plt.show()

ppn = Perceptron(epochs = 10000, lr = 0.3)

ppn.train(x,y)

# 作图
x = np.arange(0, np.pi + np.pi/100, np.pi/100)
target = np.sin(x)
y = np.zeros(101)
for i in range (1, 101):
    y[i] = ppn.predict(x[i])
plt.plot(x, y)
plt.plot(x, target)
plt.show()


x = np.arange(0, 10000, 1)
plt.plot(x, ppn.errors_)
plt.show()

#%%
