import matplotlib.pyplot
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.linalg import norm

matplotlib.pyplot.close()

# open file for read
X = np.loadtxt('Data/data4.txt')

# display data
plt.scatter(X[:, 0], X[:, 1])
plt.title('data')
plt.xlabel('X2')
plt.ylabel('X1')
plt.show()

N, K = X.shape
m = 2
k = 1
k_max = 40000

# init W
W = np.zeros((m, K))
for i in range(m):
    W[i, :] = X[i, :]

# next iteration
new_W = W.copy()
# matrix for classification
U = np.zeros((N, 2))
# study parameters
h = 0.01
eps = 1e-8

kMin = np.ceil(np.log(eps / (norm(X, 'fro') ** 2)) / np.log(1 - h))
print("Рассчетное kMin", kMin)

while k < kMin:
    index = np.random.randint(0, N)
    d = cdist(X[index, :].reshape(1, -1), W)

    pos = np.argmin(d)
    new_W[pos, :] += h * (X[index, :] - W[pos, :])

    sub = np.var(new_W - W)
    if sub < eps:
        W = new_W
        break
    W = new_W.copy()
    k += 1

print('iterations =', k)
# classify objects
for i in range(N):
    d = cdist(X[i, :].reshape(1, -1), W)

    pos = np.argmin(d)
    U[i, 0] = pos
    U[i, 1] = d[0][pos]

plt.scatter(X[:, 0], X[:, 1], c=U[:, 0])
plt.scatter(W[:, 0], W[:, 1], c='r', s=120, marker='x')
plt.title('clusters and neurons')
plt.xlabel('X2')
plt.ylabel('X1')
plt.show()
