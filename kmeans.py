import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df_train = pd.read_csv(r'C:/Users/user/downloads/dmdataset.csv')

X = df_train.iloc[:,5:33]

# reduce to 2 dimensions
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(X)



meansqerror = []
K_to_try = range(1, 6)

for i in K_to_try:
    model = KMeans(
            n_clusters=i,
            init='k-means++',
            # n_init=10,
            # max_iter=300,
            # n_jobs=-1,
            random_state=1)
    model.fit(X_pca)
    meansqerror.append(model.inertia_)

plt.plot(K_to_try, meansqerror, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('distortion')
plt.show()

#best is k=3
model = KMeans(
    n_clusters=3,
    init='k-means++',
    # n_init=10,
    # max_iter=300,
    # n_jobs=-1,
    random_state=1)

model = model.fit(X_pca)

y = model.predict(X_pca)

plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], s = 50, c = 'red', label = 'Cluster 3')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 100, c = 'blue', label = 'Centroids')
plt.title('Clusters of Students')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.legend()
plt.grid()
plt.show()

print('K Means Result : ')
print(collections.Counter(y))



# use best k=3
model_k = KMeans(
    n_clusters=3,
    init='k-means++',
    # n_init=10,
    # max_iter=300,
    # n_jobs=-1,
    random_state=1)

# fit with X instead of X_pca
model_k = model_k.fit(X)

y_final = model_k.predict(X)

print('Final K Means Result (no PCA) : ')
print(collections.Counter(y_final))




