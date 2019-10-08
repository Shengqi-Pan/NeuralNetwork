# -*- coding: utf-8 -*-
# 本文件是通过含有一个隐藏层的bp神经网络对sin(x)函数进行拟合
#%%
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, lr = 0.01, epochs = 50, decay = 0.0005):
        self.lr = lr
        self.epochs = epochs
        self.decay = decay
        # 权重初始化
        self.wi = np.random.rand(10)  # 隐藏层的权值矩阵
        self.bi = np.random.rand(10)      # 隐藏层阈值
        self.wo = np.random.rand(10)  # 输出层的权值矩阵
        self.bo = 0.3                       # 输出层阈值

    def train(self, x, y):
        # 迭代
        self.errors_ = []
        for i in range (self.epochs):
            # 初始化误差
            errors = 0
            errorsum = 0
            # 训练
            for xi, target in zip(x, y):
                # 隐藏层输入
                neth1 = xi * self.wi + self.bi
                # 隐藏层输出
                outh1 = self.sigmoid(neth1)
                # 输出层输入
                neto1 = np.inner(outh1, self.wo) + self.bo
                # 模型输出
                # outo1 = self.sigmoid(neto1)
                
                # 计算误差
                errors = 0.5 * (target - self.predict(xi)) ** 2
                errorsum += errors
                # self.errors_.append(errors)

                # de/douth1
                d1 = - (target - neto1) * self.wo
                
                # douth1/dwi
                d2 = xi * self.rsigmoid(neth1)

                # 反向传播
                # 更新输出层权值
                self.wo = self.wo - self.lr / (1 + self.decay * self.epochs) * (-(target - neto1)) * outh1               
                self.bo = self.bo - self.lr / (1 + self.decay * self.epochs) * (-(target - neto1))
                # 更新隐藏层权值
                self.wi = self.wi - self.lr / (1 + self.decay * self.epochs) * np.multiply(d2, d1)
                self.bi = self.bi - self.lr / (1 + self.decay * self.epochs) * np.multiply(self.rsigmoid(neth1), d1)
            self.errors_.append(errorsum / 9)
            print(errorsum / 9)
        return self

    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 激活函数求导
    def rsigmoid(self, x):
        return 1 * np.multiply(self.sigmoid(x), (1 - self.sigmoid(x)))

    # 第一层的输出
    def layer1out(self, x):
        return self.sigmoid(x * self.wi + self.bi)

    # 预测
    def predict(self, x):
        return np.inner(self.layer1out(x), self.wo) + self.bo

# 准备训练数据
# x = np.pi * np.random.rand(10)
x = np.arange(0, 2 * np.pi + np.pi/4, np.pi/4)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# 超参数设置
EPOCHES = 5000
LR = 0.3
DECAY = 0
# 搭网络并训练
ppn = Perceptron(epochs = EPOCHES, lr = LR, decay = DECAY)
ppn.train(x,y)

# 作图
testerror = 0
testerror_ = []
x = np.arange(0, 2 * np.pi + np.pi/180, np.pi/180)
target = np.sin(x)
y = np.zeros(361)
for i in range (0, 361):
    y[i] = ppn.predict(x[i])
    testerror += 0.5 * (y[i] - target[i]) ** 2
    testerror_.append(0.5 * (y[i] - target[i]) ** 2)

plt.plot(x, y, label = 'fit curve')
plt.plot(x, target, label = 'target curve')
plt.xlabel('input')
plt.ylabel('output')
plt.title('target curve and fit curve')
plt.legend() # 显示图例
plt.show()

# 画测试数据的误差
x = np.arange(1, 362, 1)
plt.scatter(x, testerror_, label = 'test error')
plt.xlabel('input')
plt.ylabel('test error')
plt.title('test error')
plt.legend()
plt.show()

print(testerror/361)

# 画训练中的误差变化 
x = np.arange(0, EPOCHES, 1)
plt.plot(x, ppn.errors_, label = 'training error')
plt.xlabel('index')
plt.ylabel('training error')
plt.title('training error')
plt.legend()
plt.show()

#%%
