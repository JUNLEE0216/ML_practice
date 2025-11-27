# 손글씨 데이터
from tensorflow.keras.datasets import mnist

(train_input, train_target), (test_input, test_target) = mnist.load_data()

print(train_input.shape)
print(test_input.shape)

import matplotlib.pyplot as plt
plt.imshow(train_input[800], cmap='gray')
plt.show()

# 실습 과제 : 손글씨 이미지 분류 모델 만들기
# 1. 차원 변환 및 훈련 세트, 테스트 세트 분리 작업
from sklearn.model_selection import train_test_split

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
    
# 2. 모델 구조 정의하기 (사용자의 판단에 의해 처리됨)
import keras
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

# 3. 모델 실행 환경 정의하기
# 4. 모델 실행 (콜백 적절히 정의하기)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

# 5. 그래프를 이용해 손실 점수 표현하기
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 6. 충분한 학습이 끝났다고 판단되면, predict 함수 이용한 예측 실시
import numpy as np

preds = model.predict(test_scaled[10:11])
print(np.argmax(preds))