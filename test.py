from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
 
fig = plt.figure()
ax = Axes3D(fig)
x1 = np.linspace(-10, 10, 11) / np.pi
x2 = np.linspace(-10, 10, 11) / np.pi
X1, X2 = np.meshgrid(x1, x2)#网格的创建，这个是关键
Z = np.sinc(X1) * np.sinc(X2)
X1, X2 = np.meshgrid(x1 * np.pi, x2 * np.pi)#网格的创建，这个是关键
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('input data')
# ax.scatter3D(X1, X2, Z)
ax.plot_surface(X1, X2, Z, cmap='rainbow', cstride=1, rstride=1)
plt.show()
