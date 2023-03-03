import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats import bartlett


def check_on_unit_matrix(cor_matrix):
    n = X.shape[0]
    p = X.shape[1]
    chi_sq = - (n - 1 - (2 * p + 5) / 6) * np.log(linalg.det(cor_matrix))
    df = p * (p - 1) / 2
    p_value = 1 - bartlett(chi_sq, df)
    if p_value < 0.05:
        print('Корреляционная матрица значимо отличается от единичной матрицы (p-value = {0:.4f})'.format(p_value))
    else:
        print('Корреляционная матрица не значимо отличается от единичной матрицы (p-value = {0:.4f})'.format(p_value))


# создаем исходные данные
X = np.loadtxt("Data/data4.txt", delimiter="\t")

# центрируем данные (вычитаем среднее значение)
X_centered = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# вычисляем ковариационную матрицу
cov_matrix = np.cov(X_centered.T)


check_on_unit_matrix(cov_matrix)

# вычисляем собственные значения и собственные векторы
eigen_values, eigen_vectors = linalg.eigh(cov_matrix)

# выбираем первые две главные компоненты
first_pc = eigen_vectors[:, -1]
second_pc = eigen_vectors[:, -2]

# проектируем данные на первые две компоненты
projected_1 = X_centered.dot(first_pc)
projected_2 = X_centered.dot(second_pc)

# выводим результат на одном графике
plt.axhline(y=0, color='r', linestyle='--')
plt.scatter(projected_1, projected_2, alpha=0.67)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('PCA Projection')
plt.show()

