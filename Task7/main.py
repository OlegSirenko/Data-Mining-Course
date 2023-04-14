import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Создать нейронную сеть на основе карты Кохонена (например, 2x2) (newsom).
n_clusters = 4
kohonen_map = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
# реализует алгоритм k-средних с мини-пакетами, который отличается от алгоритма обучения нейронной сети Кохонена.
# он использует другой метод обновления центроидов кластеров на основе случайно выбранных подмножеств данных.

# 2. Обучить сеть на основе карты Кохонена (net.trainParam.epochs=100, train), используя выборку вашего варианта (
# таблица 7.1). Файл данных Learning_data*.txt.
data = np.loadtxt('Data/Learning_data4.txt')
kohonen_map.fit(data)

# 3. Классифицировать исходные объекты в кластеры с использованием разработанной нейронной сети (sim, vec2ind).
clusters = kohonen_map.predict(data)

# 4. Загрузить файл данных PCA_data*.txt согласно вашему варианту (таблица 7.1). Файл содержит исходные данные в
# координатах первых двух главных компонент рассеяния Z1 и Z2 . Отобразить графически исходные данные на диаграмме
# рассеяния с учетом классификации объектов нейронной сетью (gscatter, axis).
pca_data = np.loadtxt('Data/PCA_data4.txt')
pca = PCA(n_components=2)
pca_data_transformed = pca.fit_transform(pca_data)
colors = plt.cm.get_cmap('tab10', n_clusters)
plt.scatter(pca_data_transformed[:, 0], pca_data_transformed[:, 1], c=[colors(i) for i in clusters])
plt.xlabel('Z1')
plt.ylabel('Z2')
plt.title('Кластеризация объектов')
plt.show()

# Вывод проекций объектов на оси Z1 и Z2 для каждого кластера
for cluster in np.unique(clusters):
    print(f'Кластер {cluster}:')
    cluster_data = pca_data_transformed[clusters == cluster]
    for z1, z2 in cluster_data:
        print(f'Z1: {z1}, Z2: {z2}')

# 5. Сгруппировать объекты (файл данных Learning_data*.txt) в кластеры, рассчитать средние значения по каждому
# признаку в кластерах (mean).
cluster_means = []
for cluster in np.unique(clusters):
    cluster_mean = data[clusters == cluster].mean(axis=0)
    cluster_means.append(cluster_mean)


# 6. Для каждого кластера построить график средних значений признаков объектов (характеристических векторов),
# попавших в кластер (subplot, plot, subtitle).
fig, axs = plt.subplots(len(cluster_means), 1)
for i, cluster_mean in enumerate(cluster_means):
    axs[i].plot(cluster_mean, c=colors(i))
    axs[i].set_title(f'Кластер {i + 1}')
fig.suptitle('Средние значения признаков по кластерам')
fig.text(0.5, 0.04, 'Признак', ha='center')
fig.text(0.04, 0.5, 'Среднее значение', va='center', rotation='vertical')
fig.subplots_adjust(hspace=1)  # настроить вертикальное расстояние между графиками
plt.show()

