import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('Qt5Agg')


def F(X):
    x, y = X
    A = 2
    B = 12
    return (x ** 2 + y ** 2) / A - B * (np.cos(x) + np.cos(y))


def constraint(X):
    x, y = X
    return np.array([-8 - x, x - 8, -8 - y, y - 8])


def stochastic_search(N, a, b):
    X = a + (b - a) * np.random.rand(N, 2)
    F_values = [F(x) for x in X]
    F_min = min(F_values)
    X_min = X[F_values.index(F_min)]
    return X_min, F_min


def simulated_annealing(F, x0):
    cons = ({'type': 'ineq', 'fun': constraint})
    res = basinhopping(F, x0, niter=10, T=1.0, stepsize=0.5, minimizer_kwargs={"constraints": cons})
    return res.x


N = 200
a = np.array([-8, -8])
b = np.array([8, 8])
X_min_stochastic_search, F_min_stochastic_search = stochastic_search(N, a, b)
print(f"Минимальное значение F (stochastic search): {F_min_stochastic_search}")
print(f"X_min (stochastic search): {X_min_stochastic_search}")

x0 = [0, 0]
X_min_simulated_annealing = simulated_annealing(F, x0)
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
