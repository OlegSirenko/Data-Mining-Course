import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2


def check_on_sum(x, mat_cov):
    sum_priznaki = np.sum(np.var(mat_cov, axis=0)).round(4)  # SUM_X
    sum_proekts = np.sum(np.var(x, axis=0)).round(4)
    if sum_priznaki == sum_proekts:
        print("Проверка на суммы пройденна")
    else:
        print("Проверка на суммы не пройдена")


def check_on_unit_matrix(cor_matrix, k, n):
    x_kv = chi2.ppf(0.95, k * (k - 1) / 2)
    d = 0
    for i in range(k):
        for j in range((i + 1), k):
            d = d + cor_matrix[i, j] ** 2
    d = d * n
    if d > x_kv:
        print("Корреляционная матрица значимо отличается от единичной матрицы.")
    else:
        print("Корреляционная матрица значимо отличается от единичной матрицы.")


if __name__ == "__main__":
    # создаем исходные данные
    X = np.loadtxt("Data/data4.txt", delimiter="\t")

    # центрируем данные (вычитаем среднее значение) + нормировка данных
    X_centered = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # вычисляем ковариационную матрицу
    cov_matrix = np.cov(X_centered.T)
    print(cov_matrix.shape)
    N = cov_matrix.shape[0]
    K = cov_matrix.shape[1]
    check_on_unit_matrix(cov_matrix, N, K)

    # вычисляем собственные значения и собственные векторы
    eigen_values, eigen_vectors = linalg.eig(cov_matrix)  # eigen_vectors = A, eigen_values = Lambda

    # projecting the original data on the eigenvectors space
    Z = np.asmatrix(X_centered)*eigen_vectors
    check_on_sum(X_centered, Z)

    proekts = np.var(Z, axis=0)
    sum_proects = np.sum(proekts)

    gamma = np.zeros(N)
    alpha = np.zeros(N)
    gamma[0] = proekts[0, 0]/sum_proects

    for i in range(N):
        alpha[i] = proekts[0, i]/sum_proects

    for j in range(1, N):
        gamma[j] = gamma[j-1]+alpha[j]

    print("Вектор Альфа: \n", np.round(alpha, decimals=4))
    print("Вектор Гамма: \n", np.round(gamma, decimals=4))


    # выбираем первые две главные компоненты
    first_pc = eigen_vectors[:, -1]
    second_pc = eigen_vectors[:, -2]

    # проектируем данные на первые две компоненты
    projected_1 = X_centered.dot(first_pc)
    projected_2 = X_centered.dot(second_pc)

    # выводим результат на одном графике
    plt.scatter(projected_1, projected_2, alpha=0.67)
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.grid()
    plt.title('PCA Projection')
    plt.show()
