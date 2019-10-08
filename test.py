from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
 
fig = plt.figure()
ax = Axes3D(fig)
x1 = np.linspace(-10, 10, 21) / np.pi
x2 = np.linspace(-10, 10, 21) / np.pi
X1, X2 = np.meshgrid(x1, x2)#网格的创建，这个是关键
Z = np.sinc(X1) * np.sinc(X2)
plt.xlabel('x1')
plt.ylabel('x2')
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
