from scipy.stats.distributions import chi2
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = np.loadtxt("Data/data4.txt", delimiter="\t")
    X = data  # получили матрицу 100 на 8

    N = X.shape[0]  # 100
    K = X.shape[1]  # 8

    mat = np.mean(X, axis=1)  # получили матожидание столбцов
    sigma = np.std(X, axis=1)  # получили среднеквадратичное отклонение столбцов

    print(N, K, mat.shape, sigma.shape)

    X_std = np.zeros(X.shape)  # инициализация стандартизированной матрицы, 100 на 8

    for i in range(0, N):
        for j in range(0, K):
            X_std[i][j] = (X[i][j] - mat[j]) / sigma[j]

    X_std = np.matrix(X_std)  # стандартизированная матрица

    R = (X_std.transpose() * X_std) / (N - 1)  # (8, 100) * (100,8) = (8, 8) матрица ковариации

    d = 0
    for i in range(K):
        for j in range((i + 1), K):
            d = d + R[i, j] ** 2
    d = d * N

    x_kv = chi2.ppf(0.95, K * (K - 1) / 2)
    print("Проверка на Хи^2 пройдена:", d > x_kv)

    L, A = la.eig(R)  # L -- Собственные значения, каждое из которых повторяется в соответствии со своей кратностью
    A = np.fliplr(A)  # A -- Нормализованный левый собственный вектор, соответствующий собственному значению
    L = np.flip(L)

    Z = X_std * A
    print(Z.shape)
    sum_priznaki = np.sum(np.var(X_std, axis=0))  # SUM_X
    sum_proekts = np.sum(np.var(Z, axis=0))

    print("Проверка сумм: ", sum_priznaki == sum_proekts)

    otn_d_rasbr = np.var(Z, axis=0) / sum_proekts  # относительна€ долю разброса, приход€ща€с€ на главные компоненты
    print(otn_d_rasbr)
    otn_d_rasbr_1_2 = otn_d_rasbr[:, 0] + otn_d_rasbr[:, 1]
    print(otn_d_rasbr_1_2)

    covariance = np.cov(Z)

    print(np.ndarray.flatten(Z[:, 1]))

    plt.axhline(y=0.0, color='r', linestyle='--')

    plt.scatter(covariance[:, 0], covariance[:, 1])
    plt.xlabel("Z[0]")
    plt.ylabel("Z[1]")
    plt.title("Диаграмма рассеяния для первых 2-х компонент")
    plt.show()
