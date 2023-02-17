import numpy as np

N = 5
M = 5

Q = np.random.randint(low=1, high=10, size=(N, M))
print(Q)

# _sum = 0
# for i in range(N):
#     for j in range(M):
#         _sum = _sum + Q[i][j]

main_diag = 0
poboch_diag = 0
for i in range(N):
    for j in range(M):
        if i == j and not i == int(N/2):
            main_diag += Q[i][j]
            #print(Q[i][j])

Q = np.reshape(Q, (N * N,))

j = 1
for i in range(N * N):
    if i == (N - 1) * j and j <= N:
        poboch_diag += Q[i]
        j += 1

# change this!




print(main_diag)
print(poboch_diag)
print("Сумма элементов главной и побочной диагонали:", main_diag+poboch_diag)
# print(" Проверочный результат: \n", np.sum(Q), "\n Результат по циклу: \n", _sum)
