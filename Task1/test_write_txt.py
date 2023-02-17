import numpy as np

x = np.arange(start=0, step=0.1, stop=5)
y = 2 * (x ** 2) + x - 1

M = np.append([x], [y], axis=0)
print(M)
np.savetxt("MyFile.txt", M, delimiter=',', fmt='%f')

