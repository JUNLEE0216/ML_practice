import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, None, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, None, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, None, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, None, 975.0, None]


smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, None, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, None, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

bream_df = pd.DataFrame({'length': bream_length, 'weight': bream_weight})
bream_df_clean = bream_df.dropna()

smelt_df = pd.DataFrame({'length': smelt_length, 'weight': smelt_weight})
smelt_df_clean = smelt_df.dropna()


length = bream_df_clean['length']+ smelt_df_clean['length']
weight = bream_df_clean['weight'] + smelt_df_clean['weight']

fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)

fish_target = [1]*30 + [0]*12
print(fish_target)

kn = KNeighborsClassifier()

kn.fit(fish_data, fish_target)

kn.score(fish_data, fish_target)

plt.scatter(bream_df_clean['length'], bream_df_clean['weight'])
plt.scatter(smelt_df_clean['length'], smelt_df_clean['weight'])
plt.scatter(30,600, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(kn.predict([[30, 600]]))

print(kn._fit_X)  #훈련 데이터 세트

print(kn._y)      #타깃 데이터 세트

kn42 = KNeighborsClassifier(n_neighbors=42)

kn42.fit(fish_data, fish_target)

kn42.score(fish_data, fish_target)

print(30/42)