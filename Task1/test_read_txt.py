import numpy as np
import matplotlib.pyplot as plt

M_load = np.matrix

M_load = np.loadtxt("MyFile.txt", delimiter=',')
print(M_load)

plt.plot(M_load.transpose()[:, 0], M_load.transpose()[:, 1])
plt.show()

