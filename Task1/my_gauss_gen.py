import numpy as np


def my_gauss_gen(N, M, D):
    result = np.random.randn(N, 1) * (D**0.5) + M
    return result
