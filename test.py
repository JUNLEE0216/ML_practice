import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('C:\MachineLearning\cluster_data.csv')

inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    inertia.append(km.inertia_)

import numpy as np

data = df.to_numpy()
km = KMeans(n_clusters=4)
km.fit(data)
labels = km.labels_
centroids = km.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='^', label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()