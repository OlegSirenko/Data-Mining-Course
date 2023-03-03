import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# создаем исходные данные
X = np.loadtxt("Data/data4.txt", delimiter="\t")

# центрируем данные (вычитаем среднее значение)
X_centered = X - np.mean(X, axis=0)

# вычисляем ковариационную матрицу
cov_matrix = np.cov(X_centered.T)

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
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.show()

