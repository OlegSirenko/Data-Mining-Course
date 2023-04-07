import matplotlib.pyplot
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.linalg import norm

# open file for read
X = np.loadtxt('Data/data4.txt')

N, K = X.shape
m = 2  # кол-во нейронов
k = 1  # кол-во итераций
k_max = 10000000  # максимальное кол-во итераций

# init W
# W = np.zeros((m, K))
# for i in range(m):
#     W[i, :] = X[i, :]

W = np.random.rand(m, X.shape[1])

# next iteration
new_W = W.copy()
# matrix for classification
U = np.zeros((N, 2))
# study parameters
h = 0.25
eps = 1e-6

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(W[:, 0], W[:, 1], c='r', s=120, marker='x')
plt.title('data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

kMin = np.ceil(np.log(eps / (norm(X, 'fro') ** 2)) / np.log(1 - h))
print("Рассчетное kMin", kMin)

while k < k_max:
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
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
