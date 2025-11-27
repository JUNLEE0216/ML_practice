import pandas as pd 

df = pd.read_csv("C:\MachineLearning\cluster_data.csv")
df.head()

# 1. 데이터 읽어들이기
df.info()

# 2. 군집 개수를 구해야 한다 (이너셔, 엘보우)
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 8):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    inertia.append(km.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1, 8), inertia)

# 3. K평균 모델을 생성하여 학습 및 결과 도출
import numpy as np

data = df.to_numpy()
km = KMeans(n_clusters=5)
km.fit(data)
print(np.unique(km.labels_, return_counts=True))
centroids = km.cluster_centers_
# 4. 결과를 시각화 (라벨링된 결과를 포진시켜봐~~)
plt.cla()
plt.scatter(data[:, 0], data[:, 1], c=km.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='^', label='Centroids')
plt.show()

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
seq = tokenizer.texts_to_sequences(sentences)
