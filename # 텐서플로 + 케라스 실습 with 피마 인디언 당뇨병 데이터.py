# 텐서플로 + 케라스 실습 with 피마 인디언 당뇨병 데이터 
# 시각화를 위한 데이터프레임 변환 및 조회 (사실은 선택사항)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/diabetes.csv')
#  df["Outcome"].value_counts() # 인원수 파악... 500:268

fig, axs = plt.subplots(figsize=(16,8) , ncols=5 , nrows=2)
features = df.columns

for i , feature in enumerate(features):
    row = int(i/5)
    col = i % 5
    sns.regplot(x=feature , y='Outcome', data=df, ax=axs[row][col])

plt.show()

# 훈련 및 검증 세트를 분리한다 (전후로 전처리 고려)
from sklearn.model_selection import train_test_split
data = df.iloc[:,0:8]
target = df.iloc[:,8]
train_input , test_input , train_target , test_target = train_test_split(data, target, test_size=0.2, random_state=18)

# 훈련 세트로 훈련하고, 검증 세트로 점수 매기고, 예측도 해보고 
from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, input_shape=(8,), activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = keras.callbacks.EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True)
history = model.fit(train_input, train_target, batch_size=16,epochs=500, validation_split=0.2, callbacks=[es])

score = model.evaluate(train_input, train_target)
print(score)
model.summary()