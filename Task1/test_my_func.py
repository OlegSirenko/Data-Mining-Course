import numpy as np
#import scipy
from my_func import my_func
import matplotlib.pyplot as plt


#Задание диапазона изменения X
X_left = -2
X_right = 2
# Задание диапазона изменения Y
Y_left = -3
Y_right = 3
# Задание количества сгенерированных точек
N = 1000
# Вызов функции
X, Y = my_func(X_left, X_right, Y_left, Y_right, N)
# Построение графика функции
plt.plot(X, Y, 'o')
plt.show()


BinNumber = 20  # Инициализация гистограммы
k = np.arange(start=0, stop=BinNumber+1, step=1)
X_bins = X_left + k*(X_right - X_left)/BinNumber  # Вычисление границ карманов на оси X
Y_bins = Y_left + k*(Y_right - Y_left)/BinNumber


plt.hist(X, X_bins, edgecolor='k')
plt.title("X_Bins")
plt.xlabel("X")
plt.ylabel("Occurrences")
plt.show()
plt.hist(Y, Y_bins, edgecolor='k')
plt.title("Y_bins")
plt.xlabel("Y")
plt.ylabel("occurrences")
plt.show()



