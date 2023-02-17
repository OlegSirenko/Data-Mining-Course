import numpy as np

def vrosenbrock(x: np.array, y: np.array):
    return 100 * (y - x**2)**2 + (1 - x) ** 2
