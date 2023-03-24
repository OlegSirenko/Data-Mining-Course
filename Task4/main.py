import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import single, cophenet
from scipy.cluster.hierarchy import ward, fcluster

X = np.loadtxt("Data/data4.txt")

x = X[:, 0]
y = X[:, 1]
N = X.shape[0]
K = X.shape[1]

print(N, K)
plt.plot(x, y, "*")
plt.title("Входные данные")
plt.show()

# Евклидово
# метрики
d_eucl = pdist(X, 'euclidean')
# методы
link_eucl_single = linkage(d_eucl, "single")
link_eucl_averag = linkage(d_eucl, 'average')
link_eucl_median = linkage(d_eucl, 'median')

# Минковского
# Расстояние
d_mink = pdist(X, 'minkowski', p=4.)
# Метрики
link_mink_single = linkage(d_mink, "single")
link_mink_averag = linkage(d_mink, 'average')
link_mink_median = linkage(d_mink, 'median')
# Чебышева
# Расстояние
d_cheb = pdist(X, 'chebyshev')
# Метрики
link_cheb_single = linkage(d_cheb, "single")
link_cheb_averag = linkage(d_cheb, "average")
link_cheb_median = linkage(d_cheb, 'median')
#

coph_corr_coeff = np.zeros(shape=(3, 3))

coph_corr_coeff[0, 0] = cophenet(link_eucl_single, d_eucl)[0]
coph_corr_coeff[1, 0] = cophenet(link_eucl_averag, d_eucl)[0]
coph_corr_coeff[2, 0] = cophenet(link_eucl_median, d_eucl)[0]
#
coph_corr_coeff[0, 1] = cophenet(link_mink_single, d_mink)[0]
coph_corr_coeff[1, 1] = cophenet(link_mink_averag, d_mink)[0]
coph_corr_coeff[2, 1] = cophenet(link_mink_median, d_mink)[0]

coph_corr_coeff[0, 2] = cophenet(link_cheb_single, d_cheb)[0]
coph_corr_coeff[1, 2] = cophenet(link_cheb_averag, d_cheb)[0]
coph_corr_coeff[2, 2] = cophenet(link_cheb_median, d_cheb)[0]
print(coph_corr_coeff)

print("Евклидово Среднее:\n", np.max(coph_corr_coeff))
print("\nЧебышева ближнего:\n", np.min(coph_corr_coeff))

plt.figure()
dendrogram(link_eucl_averag)
plt.title('(Лучшее) Евклидова метрика + метод среддней связи')
plt.show()

plt.figure()
dendrogram(link_cheb_single)
plt.title('(Худшее) Чебышева метрика + метод ближней связи')
plt.show()

N_clust = 5
clust = fcluster(link_eucl_averag, N_clust, 'maxclust')

plt.figure()
plt.scatter(x, y, c=clust)
#plt.show()

clust_centr = np.zeros((N_clust, K))  # матрица x и y координат центров
# расстояния между центрами кластеров
rast_betw_clust = np.zeros((N_clust, N_clust))
# радиусы кластеров
clust_rad = np.zeros((N_clust, 1))
# дисперсии кластеров
clust_disp = np.zeros((N_clust, 1))
print(clust)
for k in range(0, N_clust):
    print("№", k)
    obj = np.argwhere(clust == k + 1)
    if obj is None:
        continue
    N_obj = len(obj)
    print(obj)
    singl_clust = np.zeros((N_obj, K))
    for i in range(0, N_obj):
        singl_clust[i, :] = X[obj[i], :]
        clust_centr[k, :] = clust_centr[k, :] + singl_clust[i, :]

    clust_centr[k, :] = clust_centr[k, :] / N_obj
    clust_rast_centr = np.zeros((N_obj, 1))
    for i in range(0, N_obj):
        for l in range(0, K):
            clust_rast_centr[i] = clust_rast_centr[i] + (singl_clust[i, l] - clust_centr[k, l]) ** 2
        clust_disp[k] = clust_disp[k] + clust_rast_centr[i]
        clust_rast_centr[i] = np.sqrt(clust_rast_centr[i])

    clust_disp[k] = clust_disp[k] / N_obj
    print(clust_rast_centr)
    clust_rad[k] = max(clust_rast_centr)

for i in range(0, N_clust):
    for j in range(i + 1, N_clust):
        for l in range(0, K):
            rast_betw_clust[i, j] = rast_betw_clust[i, j] + (clust_centr[i, l] - clust_centr[j, l]) ** 2
        rast_betw_clust[i, j] = np.sqrt(rast_betw_clust[i, j])

print('Расстояния между центрами кластеров')
print(rast_betw_clust)

plt.scatter(clust_centr[:, 0], clust_centr[:, 1], 20, 'k')
t = np.arange(start=0, step=np.pi / 180, stop=2 * np.pi)
for i in range(0, N_clust):
    x = clust_rad[i] * np.cos(t) + clust_centr[i, 0]
    y = clust_rad[i] * np.sin(t) + clust_centr[i, 1]
    plt.plot(x, y, 'k')

plt.title('Найденные кластеры и их центры')
plt.xlabel('X2')
plt.ylabel('X1')
plt.legend()
plt.show()

# 1. Название: Иерархические методы кластерного анализа.
#
# 2. Суть метода: Иерархический кластерный анализ является одним из способов разбиения данных на группы. Этот метод
# рассматривает данные как иерархическую структуру, где каждый элемент начинает отдельной группой. Затем группы
# объединяются, пока конечно не получится одна большая группа. В процессе объединения групп учитывается степень их
# схожести, что позволяет находить скрытые закономерности в данных.
#
# 3. Что получится в итоге: В результате работы иерархического кластерного анализа получается дендрограмма,
# на которой видна структура данных. Каждый уровень дерева представляет собой группу объектов, а расстояние между
# уровнями отражает степень схожести объектов в них. Информация об устройстве дерева позволяет определить число
# групп, на которые можно разбить данные.
#
# 4. Как проверить правильность итоговых данных: Для проверки правильности результатов необходимо применить оценку
# качества кластеризации. Одним из способов является метод "Silhouette index", который оценивает расстояние между
# объектами внутри кластера и между кластерами. Чем ближе значение индекса к 1, тем лучше кластеризация. Кроме того,
# можно визуализировать группы на графике, чтобы оценить их взаимное расположение и схожесть.
#
#
