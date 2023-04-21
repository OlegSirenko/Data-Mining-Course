import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('Qt5Agg')

def F(X):
    x, y = X
    A = 2
    B = 12
    return (x ** 2 + y ** 2) / A - B * (np.cos(x) + np.cos(y))


def stochastic_search(N, a, b):
    X = a + (b - a) * np.random.rand(N, 2)
    F_values = [F(x) for x in X]
    F_min = min(F_values)
    X_min = X[F_values.index(F_min)]
    return X_min, F_min


def simulated_annealing(F, x0, bounds, T=50.0, T0=0.001, v=0.99):
    X = x0
    l = 0
    while T > T0:
        l += 1
        z = np.random.normal(size=len(X))
        X_new = X + z * T
        for i in range(len(X)):
            if X_new[i] < bounds[i][0]:
                X_new[i] = bounds[i][0]
            elif X_new[i] > bounds[i][1]:
                X_new[i] = bounds[i][1]
        dE = F(X_new) - F(X)
        if dE < 0:
            X = X_new
        else:
            P = np.exp(-dE / T)
            if np.random.uniform() < P:
                X = X_new
            else:
                T = v * T
    return X


N = 1000
a = np.array([-8, -8])
b = np.array([8, 8])
X_min_stochastic_search, F_min_stochastic_search = stochastic_search(N, a, b)
print(f"Минимальное значение F (stochastic search): {F_min_stochastic_search}")
print(f"X_min (stochastic search): {X_min_stochastic_search}")

x0 = [0, 0]
bounds = [(-8, 8), (-8, 8)]
X_min_simulated_annealing = simulated_annealing(F, x0, bounds)
F_min_simulated_annealing = F(X_min_simulated_annealing)
print(f"Минимальное значение F (simulated annealing): {F_min_simulated_annealing}")
print(f"X_min (simulated annealing): {X_min_simulated_annealing}")

x = np.linspace(-8, 8, 100)
y = np.linspace(-8, 8, 100)
X, Y = np.meshgrid(x, y)
Z = F([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(X_min_stochastic_search[0], X_min_stochastic_search[1], F_min_stochastic_search, c='r',
           label='stochastic search')
ax.scatter(X_min_simulated_annealing[0], X_min_simulated_annealing[1], F_min_simulated_annealing, c='b',
           label='simulated annealing')
ax.legend()
plt.show()
