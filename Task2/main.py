from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def dover_int(t1, t2, C, q):
    b = len(q)
    ci_form = np.zeros((2, 2))
    if b == 2:
        for i in range(1):
            for j in range(1):
                if j == 1:
                    ci_form[i][j] = q[i] + t1 * np.sqrt(C[i])
                else:
                    ci_form[i][j] = q[i] + t2 * np.sqrt(C[i])
    else:
        for i in range(1):
            if i == 0:
                ci_form[i] = q + np.dot(np.sqrt(C), t1)
            else:
                ci_form[i] = q + np.dot(np.sqrt(C), t2)
    return ci_form


def autocorrelation(r_in, k_in, n):
    res = np.zeros(k_in)
    sum1 = np.sum(r_in ** 2) / n
    for i in range(k_in):
        temp = 0
        for j in range(n - i):
            temp = temp + r_in[j] * r_in[j + i]

        res[i] = 1. / (n - i) * temp / sum1
    return res


# Законы распределения
def gamma_raspred(x_in, a, b):
    result = (x_in ** a * np.exp(-x / b)) / (b ** (a + 1) * gamma(a + 1))
    return result


def reley_raspred(x_in, sigma_in):
    result = (x_in / sigma_in ** 2) * np.exp(-x_in ** 2 / (2 * (sigma_in ** 2)))
    return result


def weibull_raspred(x_in, a, b):
    result = a * b * x_in ** (b - 1) * np.exp(-a * x_in ** b)
    return result


#################


data = np.loadtxt("Data_Lab2/data4_MINE_VAR.txt", delimiter=" ")
x = data[:, 0]
y = data[:, 1]
sigma = data[:, 2]

plt.plot(x, y, "-", label="Эксперементальные данные")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("y(x)")

# НЕЛИНЕЙНЫЙ МЕТОД НАИМЕНЬШИХ
# Функция 7
params_gamma, cov_3, infodict, mesg, ier = curve_fit(f=gamma_raspred, xdata=x, ydata=y, bounds=(0, 3), full_output=True)
print("Параметры для функции 7: \n", params_gamma)
y_gamma = gamma_raspred(x, *params_gamma)
plt.plot(x, gamma_raspred(x, *params_gamma), 'r-', label='Гамма-распределение')
plt.legend()

# Функция 4
params_weilbull, cov_4 = curve_fit(f=weibull_raspred, xdata=x, ydata=y, bounds=(0, 3))
#print(params_weilbull)
y_weil = weibull_raspred(x, *params_weilbull)
plt.plot(x, weibull_raspred(x, *params_weilbull), 'g-', label='Распределение Вейбулла')
plt.legend()

# Функция 3
params_reley, cov_7 = curve_fit(f=reley_raspred, xdata=x, ydata=y, bounds=(2, 5))
#print(params_reley)
y_rel = reley_raspred(x, *params_reley)
plt.plot(x, reley_raspred(x, *params_reley), 'y-', label='Распределение Реллея')
plt.legend()
plt.show()

chi2 = {3: 0, 4: 0, 7: 0}

for i in range(len(y)):
    chi2[3] += ((y_rel[i] - y[i]) / sigma[i]) ** 2
    chi2[4] += ((y_weil[i] - y[i]) / sigma[i]) ** 2
    chi2[7] += ((y_gamma[i] - y[i]) / sigma[i]) ** 2
#print("Chi^2: ")
#print(chi2[3], chi2[4], chi2[7])

chi2[3] = chi2[3] / (200 - 1 - 1)
chi2[4] = chi2[4] / (200 - 2 - 1)
chi2[7] = chi2[7] / (200 - 2 - 1)
print("Chi^2 нормированное: ")
print(chi2[3], chi2[4], chi2[7])

r = {3: (y - y_rel) / sigma, 4: (y - y_weil) / sigma, 7: (y - y_gamma) / sigma}
print()

# plt.plot(x, r[3], label="Остатки функции 3")
# plt.plot(x, r[4], label="Остатки функции 4")
plt.axhline(y=0.0, color='r', linestyle='--')
plt.plot(x, r[7], label="Остатки функции 7")
plt.xlabel("x")
plt.ylabel("r")
plt.legend()
plt.show()

N = int(len(y) / 2)
k = np.arange(start=0, stop=N)

AC = {3: autocorrelation(r[3], N, len(y)), 4: autocorrelation(r[4], N, len(y)), 7: autocorrelation(r[7], N, len(y))}
# plt.plot(k, AC[3], label="Автокорелляция для 3")
# plt.plot(k, AC[4], label="Автокорелляция для 4")

plt.axhline(y=0.0, color='r', linestyle='--')
plt.plot(k, AC[7], label="Автокорелляция для 7")
plt.legend()
plt.xlabel("k")
plt.ylabel("AC")
plt.show()

N = len(y)
t1 = stats.t.rvs(0.16, N-1)
t2 = stats.t.rvs(1 - 0.16, N-1)

C = {3: np.diag(cov_3), 4: np.diag(cov_4), 7: np.diag(cov_7)}

ci_f = {3: dover_int(t1, t2, C[3], params_reley),
        4: dover_int(t1, t2, C[4], params_weilbull),
        7: dover_int(t1, t2, C[7], params_gamma)
        }
for i in [3, 4, 7]:
    print(ci_f[i])
