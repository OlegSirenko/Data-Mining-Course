from my_gauss_gen import my_gauss_gen
import matplotlib.pyplot as plt
import numpy as np

N = 10**6
D = 30
M = 20

X = my_gauss_gen(N, M, D)
BinNumber = 40
k = np.arange(start=0, stop=BinNumber + 1, step=1)

X_left = np.min(X)
X_right = np.max(X)
X_bins = X_left + k * (X_right - X_left) / BinNumber
plt.hist(X, X_bins, edgecolor='k')
plt.title("x bins")
plt.xlabel("x")
plt.ylabel("occurrences")
plt.show()
