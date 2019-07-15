import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import collections
from sklearn.decomposition import PCA

df_train = pd.read_csv(r'C:/Users/user/downloads/dmdataset.csv')
X = df_train.iloc[:,5:33]

#without pca
bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=5820)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)
#3clusters
y = ms.predict(X)
print(collections.Counter(y))
#output:Counter({0: 4456, 1: 1363, 2: 1})


#with pca
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(X)
bandwidth = estimate_bandwidth(X_pca, quantile=0.3, n_samples=5820)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_pca)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)
#2clusters
y = ms.predict(X_pca)
print(collections.Counter(y))
Counter({0: 4468, 1: 1352})


#for pca plot
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X_pca[my_members, 0], X_pca[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#mean and stddeviation of clusters only for pca

y_final = pd.DataFrame(y, columns=['cluster'])
raw_result = pd.concat([X, y_final], axis=1)

y = pd.DataFrame(y, columns=['cluster'])
mean_by_student_1 = raw_result[raw_result['cluster']==0].iloc[:, 0:28].mean(axis = 1)

mean_by_student_2 = raw_result[raw_result['cluster']==1].iloc[:, 0:28].mean(axis = 1)
print('Mean cluster 1 : ' + str(mean_by_student_1.mean()) + ',STD :' + str(mean_by_student_1.std()))
print('Mean cluster 2 : ' + str(mean_by_student_2.mean()) + ',STD :' + str(mean_by_student_2.std()))
#output
#Mean cluster 1 : 3.693582938994754,STD :0.7472712892223174
#Mean cluster 2 : 1.5092455621301768,STD :0.5142820734247068
