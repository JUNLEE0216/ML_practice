import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)


km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

def draw_fruits(arr, ratio=1):
    n = len(arr)
    
    rows = int(np.ceil(n/10))
    
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
    
fruits[km.labels_ == 0]
fruits[km.labels_ == 1]
fruits[km.labels_ == 2]

km.cluster_centers_.reshape(-1, 100, 100)

inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()