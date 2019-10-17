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
        self.neurons = 10 # 神经元个数
        # 权重初始化
        self.wi = np.random.rand(self.neurons)  # 隐藏层的权值矩阵
        self.bi = np.random.rand(self.neurons)      # 隐藏层阈值
        self.wo = np.random.rand(self.neurons)  # 输出层的权值矩阵
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
                
                # 计算误差
                errors = 0.5 * (target - neto1) ** 2
                errorsum += errors

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
x = np.arange(0, 2 * np.pi + np.pi/4, np.pi/4)
y = np.sin(x)
plt.subplot(221)
plt.plot(x, y)
plt.title('input data')
plt.xlabel('input')
plt.ylabel('output')

# 超参数设置
EPOCHES = 10000
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

plt.subplot(222)
plt.plot(x, y, label = 'fit curve')
plt.plot(x, target, label = 'target curve')
plt.xlabel('input')
plt.ylabel('output')
plt.title('target curve and fit curve')
plt.legend() # 显示图例

# 画测试数据的误差
x = np.arange(1, 362, 1)
plt.subplot(223)
plt.scatter(x, testerror_, label = 'test error')
plt.xlabel('input')
plt.ylabel('test error')
plt.title('test error')
plt.legend()

print(testerror/361)

# 画训练中的误差变化 
x = np.arange(0, EPOCHES, 1)
plt.subplot(224)
plt.plot(x, ppn.errors_, label = 'training error')
plt.xlabel('index')
plt.ylabel('training error')
plt.title('training error')
plt.legend()
plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距
fig = plt.gcf()
fig.suptitle('lr=%.2f, neurons=%d, epoches=%d\ntest error=%e, training error=%e'%(ppn.lr, ppn.neurons, ppn.epochs, testerror/361, ppn.errors_[-1]),fontsize=16,fontweight=600,x=0.5,y=0.99,linespacing=1.8)
fig.set_size_inches(12, 8)
fig.savefig('task1result//lr=%.2fneurons=%depochs=%dtesterror=%etrainingerror=%e.png'%(ppn.lr, ppn.neurons, ppn.epochs, testerror/361, ppn.errors_[-1]), dpi=500)
plt.show()

#%%
