import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Load data from file
X = np.loadtxt('Data/data4.txt')

# Extract values for x, y, N, K
x = X[:, 0]
y = X[:, 1]
N, K = X.shape

# Create subplots for each graph
fig, axs = plt.subplots(1, 2, figsize=(8, 8), sharex=True, layout='constrained', sharey=True)

# Create scatter plot for original data
axs[0].scatter(x, y, s=35)
axs[0].set_title("Экспериментальные данные")

# Define initial values for k and C matrix
k = 2
C = np.zeros((k, K))
for i in range(k):
    C[i, :] = X[i, :]

# Initialize U, Q_pred, eps, m, clust_rad
U = np.zeros((N, 2))
Q_pred = 1435
eps = 0.0001435
m = 1
clust_rad = np.zeros(k)

# Begin while loop - keep looping until convergence
while True:
    # Define R - the euclidean distances between each point and each cluster center
    R = np.zeros(k)
    for i in range(N):
        for n in range(k):
            R[n] = pdist([X[i, :], C[n, :]], 'euclidean')
        n = np.argmin(R)
        U[i, 0] = n
        U[i, 1] = R[n]

    # Initialize Q_m and QQ, and iterate through each cluster
    Q_m = 0
    QQ = 0
    for p in range(k):
        # Define Objects as all datapoints in the current cluster
        Obj = np.where(U[:, 0] == p)[0]
        s = Obj.size
        # Initialize singl_clust and add euclidean distances to it for each point in current cluster
        singl_clust = np.zeros(s)
        for i in range(s):
            singl_clust[i] = U[Obj[i], 1]
            # Add euclidean distance to total cluster distance (Q_m)
            Q_m += U[Obj[i], 1]
        # Set cluster radius as maximum distance in cluster
        clust_rad[p] = np.max(singl_clust)
        # Add total cluster distance to QQ
        QQ += Q_m

    # If QQ is close enough to Q_pred, break out of loop
    if np.abs(QQ - Q_pred) <= eps:
        break
    else:
        # For each cluster, define Objects as all datapoints in the current cluster
        for l in range(k):
            Obj = np.where(U[:, 0] == l)[0]
            s = Obj.size
            # Calculate the mean of all points in current cluster, and store in C
            for j in range(K):
                summa = 0
                for i in range(s):
                    summa += X[Obj[i], j]
                C[l, j] = summa / s
        # Set Q_pred as current QQ, and increase count of iterations (m)
        Q_pred = QQ
        m += 1

# Define custom colormap
my_cmap = plt.colormaps['Set1']


# Create scatter plot with each cluster represented by different color
axs[1].scatter(x, y, c=U[:, 0], cmap=my_cmap, s=15)
axs[1].set_title('Найденные кластеры и их центры')

# Define t values for plotting ovals
t = np.arange(0, 2 * np.pi, np.pi / 180)
# For each cluster, plot an oval
for i in range(k):
    # Define x and y coordinates for oval using cluster radius and center coordinates
    x = clust_rad[i] * np.cos(t) + C[i, 0]
    y = clust_rad[i] * np.sin(t) + C[i, 1]
    axs[1].plot(x, y, 'k')

# Plot cluster centers as filled circles
axs[1].scatter(C[:, 0], C[:, 1], c='k', s=20, marker='o')
# Display both subplots
fig.supxlabel('X2')
fig.supylabel('X1')
plt.show()
