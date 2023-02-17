import numpy as np

# Задание 5

V = np.matrix([1, 2, 3, 4, 11])
M = np.matrix([[1, 2, 3, 4, 5], [1, 2, -4, 4, 5], [1, 2, 3, 4, 5]])

print("Изначальная матрица М:\n", M, "\nИзначальный вектор V:\n", V, "\n\n")

M = np.append(M, V, axis=0)
M = np.append(M, V, axis=0)

#
print("Добавление строки к матрице\n", M)
M = np.append(M, V.transpose(), axis=1)
print("Добавление столбца\n", M)
# print("Размер матрицы", M.shape)
print("Max=", np.max(M), "Min=", np.min(M))

print("Удаление вектора\n", np.delete(M, V[:, 0], axis=1))

# Задание 6
print("--------Задание 6-------")

M = np.asmatrix(np.random.rand(5, 5))
print(M)

for i in range(0, 5):
    print(f"Min по вертикали {i} = ", np.matrix.min(M[:, i]))
    print(f"Max по вертикали {i} = ", np.matrix.max(M[:, i]))
    print()

for i in range(0, 5):
    print(f"Min по горизонтали {i} = ", M[i, :].min())
    print(f"Max по горизонтали {i} = ", M[i, :].max())
    print()

print("Min по всей матрице = ", M.min(), "\nМах по всей матрице = ", M.max())

# Задание 7

print("---Задание 7-----")
A = np.matrix('[2 1; 3 4]')
B = np.matrix('[4; 11]', dtype=int)

x = np.linalg.inv(A) * B
print(np.invert(A), "\n", B)
# x = np.linalg.solve(A, B)
check = A * x - B
check = np.round(check, 8)
print("Решение Алгебраического уравнения -- \n", x)
print("Проверка ==\n", check)
