# -*- coding: utf-8 -*-
# 本文件是通过含有一个隐藏层的bp神经网络对sin(x)函数进行拟合
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

class Perceptron(object):

    def __init__(self, lr = 0.01, epochs = 50, decay = 0.0005):
        self.lr = lr
        self.epochs = epochs
        self.decay = decay
        self.neurons = 30 # 神经元个数
        # 权重初始化
        self.wi1 = np.random.randn(self.neurons)      # 连接x1隐藏层的权值矩阵
        self.wi2 = np.random.randn(self.neurons)      # 连接x2隐藏层的权值矩阵
        self.bi = np.random.randn(self.neurons)       # 隐藏层阈值
        self.wo = np.random.randn(self.neurons)       # 输出层的权值矩阵
        self.bo = 0.3                    # 输出层阈值

    def train(self, x1, x2, y):
        # 迭代
        self.errors_ = []
        m_wo = 0
        m_bo = 0
        m_wi1 = 0
        m_wi2 = 0
        m_bi = 0
        u = 0.5
        for i in range (self.epochs):
            # 初始化误差
            errors = 0
            errorsum = 0
            # 训练
            for xi1, xi2, target in zip(x1, x2, y):
                # 隐藏层输入
                neth1 = xi1 * self.wi1 + xi2 * self.wi2 + self.bi
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
                rsigmoidneth1 = self.rsigmoid(neth1)
                d21 = xi1 * rsigmoidneth1
                d22 = xi2 * rsigmoidneth1

                # 反向传播
                # 更新输出层权值
                g_bo = (-(target - neto1))
                g_wo = g_bo * outh1
                m_wo = u * m_wo + (1 - u) * g_wo
                m_bo = u * m_bo + (1 - u) * g_bo
                self.wo = self.wo - self.lr / (1 + self.decay * self.epochs) * m_wo           
                self.bo = self.bo - self.lr / (1 + self.decay * self.epochs) * m_bo
                
                # 更新隐藏层权值
                g_wi1 = np.multiply(d21, d1)
                g_wi2 = np.multiply(d22, d1)
                g_bi = np.multiply(self.rsigmoid(neth1), d1)
                m_wi1 = u * m_wi1 + (1 - u) * g_wi1
                m_wi2 = u * m_wi2 + (1 - u) * g_wi2
                m_bi = u * m_bi + (1 - u) *g_bi
                self.wi1 = self.wi1 - self.lr / (1 + self.decay * self.epochs) * m_wi1
                self.wi2 = self.wi2 - self.lr / (1 + self.decay * self.epochs) * m_wi2
                self.bi = self.bi - self.lr / (1 + self.decay * self.epochs) * m_bi
            self.errors_.append(errorsum / 121)
            print(errorsum / 121)
        return self

    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 激活函数求导
    def rsigmoid(self, x):
        return np.multiply(self.sigmoid(x), (1 - self.sigmoid(x)))

    # 第一层的输出
    def layer1out(self, x1, x2):
        return self.sigmoid(x1 * self.wi1 + x2 * self.wi2 + self.bi)

    # 预测
    def predict(self, x1, x2):
        return np.inner(self.layer1out(x1, x2), self.wo) + self.bo

# 准备训练数据
x1 = np.linspace(-10, 10, 11) / np.pi
x2 = np.linspace(-10, 10, 11) / np.pi
tmp = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))]).T    # 求x1和x2的笛卡尔积
x1 = tmp[0]
x2 = tmp[1]
y = np.sinc(x1) * np.sinc(x2)

# 画训练数据图
x1_ = np.linspace(-10, 10, 11) / np.pi
x2_ = np.linspace(-10, 10, 11) / np.pi
X1, X2 = np.meshgrid(x1_, x2_)#网格的创建，这个是关键
Z = np.sinc(X1) * np.sinc(X2)

# 超参数设置
EPOCHES = 10000
LR = 0.1
DECAY = 0.0001
# 搭网络并训练
ppn = Perceptron(epochs = EPOCHES, lr = LR, decay = DECAY)
begintimer = time()
ppn.train(x1, x2, y)
endtimer = time()
print(endtimer - begintimer)

# 作效果图
testerror = 0
testerror_ = []
x1 = np.linspace(-10, 10, 21) / np.pi
x2 = np.linspace(-10, 10, 21) / np.pi
X1, X2 = np.meshgrid(x1, x2)        #网格的创建，这个是关键
target = np.sinc(X1) * np.sinc(X2)

# 计算预测结果的矩阵并作预测图
Z = np.zeros((21, 21))
for i in range (0, 21):
    for j in range (0, 21):
        Z[i, j] = ppn.predict(X1[i, j], X2[i, j])
        testerror += 0.5 * (Z[i, j] - target[i, j]) ** 2
fig1 = plt.figure()
ax = Axes3D(fig1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('target surface and fit dot')
X1 = X1 * np.pi
X2 = X2 * np.pi
ax.scatter3D(X1, X2, Z)
ax.plot_surface(X1, X2, target, rstride=1, cstride=1, cmap='rainbow')
# plt.show()


# 画拟合结果
fig2 = plt.figure()
ax = Axes3D(fig2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('fit surface')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')


# 画测试数据的误差
testerror_ = 0.5 * np.multiply((target - Z), (target - Z))
print(testerror/441)
fig3 = plt.figure()
ax = Axes3D(fig3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('test error')
ax.plot_surface(X1, X2, testerror_, rstride=1, cstride=1, cmap='rainbow')


# 画训练中的误差变化 
fig4 = plt.figure()
x = np.arange(0, EPOCHES, 1)
plt.plot(x, ppn.errors_, label = 'training error')
plt.xlabel('index')
plt.ylabel('training error')
plt.title('training error')
plt.legend()
# fig = plt.gcf()
fig1.savefig('task2result//1lr=%.2fneurons=%depochs=%dtesterror=%etrainingerror=%e.png'%(ppn.lr, ppn.neurons, ppn.epochs, testerror/361, ppn.errors_[-1]), dpi=500)
fig2.savefig('task2result//2lr=%.2fneurons=%depochs=%dtesterror=%etrainingerror=%e.png'%(ppn.lr, ppn.neurons, ppn.epochs, testerror/361, ppn.errors_[-1]), dpi=500)
fig3.savefig('task2result//3lr=%.2fneurons=%depochs=%dtesterror=%etrainingerror=%e.png'%(ppn.lr, ppn.neurons, ppn.epochs, testerror/361, ppn.errors_[-1]), dpi=500)
fig4.savefig('task2result//4lr=%.2fneurons=%depochs=%dtesterror=%etrainingerror=%e.png'%(ppn.lr, ppn.neurons, ppn.epochs, testerror/361, ppn.errors_[-1]), dpi=500)
plt.show()

#%%