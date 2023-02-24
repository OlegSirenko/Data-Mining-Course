import math
from scipy.optimize import curve_fit
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt


# TODO: ### **For task 2**
#  - [] Make 4)
#  - [] Make 5)


def gamma_raspred(x_in, a, b):
    result = (x_in ** a * np.exp(-x / b)) / (b ** (a + 1) * gamma(a + 1))
    return result


def reley_raspred(x_in, sigma_in):
    result = (x_in / sigma_in ** 2) * np.exp(-x_in ** 2 / (2 * (sigma_in ** 2)))
    return result


def weibull_raspred(x_in, a, b):
    result = a * b * x_in ** (b - 1) * np.exp(-a * x_in ** b)
    return result


data = np.loadtxt("Data_Lab2/data4_MINE_VAR.txt", delimiter=" ")
x = data[:, 0]
y = data[:, 1]
sigma = data[:, 2]

plt.plot(x, y, "-", label="Эксперементальные данные")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("y(x)")

# Функция 7
params_gamma, cov, infodict, mesg, ier = curve_fit(f=gamma_raspred, xdata=x, ydata=y, bounds=(0, 3), full_output=True)
print("dict:\n", ier)
print(params_gamma)
plt.plot(x, gamma_raspred(x, *params_gamma), 'r-', label='Гамма-распределение')
plt.legend()

# Функция 4
params_wailbull, cov = curve_fit(f=weibull_raspred, xdata=x, ydata=y, bounds=(0, 3))
print(params_wailbull)
plt.plot(x, weibull_raspred(x, *params_wailbull), 'g-', label='Распределение Вейбулла')
plt.legend()

# Функция 3
params_reley, cov = curve_fit(f=reley_raspred, xdata=x, ydata=y, bounds=(2, 5))
print(params_reley)
plt.plot(x, reley_raspred(x, *params_reley), 'y-', label='Распределение Реллея')
plt.legend()
plt.show()
