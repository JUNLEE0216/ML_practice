import numpy as np 
import matplotlib.pyplot as plt


data = np.load(r'C:\MachineLearning\x_gray.npy')
target = np.load(r'C:\MachineLearning\y.npy')

print(data.shape, target.shape)

data_scaled = data.reshape(-1, 64, 64, 1) / 255.0

from sklearn.model_selection import train_test_split
train_scaled, test_scaled, train_target, test_target = \
train_test_split(data_scaled, target, test_size=0.2, random_state=23)

print(train_scaled.shape, train_target.shape)
print(test_scaled.shape, test_target.shape)

from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu',
                              padding='same', input_shape=(64, 64, 1)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu',
                              padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(24, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5',
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_split=0.3,
                    callbacks=[checkpoint_cb, early_stopping_cb])
                    
                    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

pred = model.predict(test_scaled)
test_predict = np.round(pred.flatten())

from sklearn.metrics import accuracy_score
accuracy_score(test_target, test_predict)

import random

check_index = random.randint(0, len(test_predict) - 1)

print(f"{check_index}번 인덱스의 예측 결과는 ", end="")
print("고양이" if test_predict[check_index] == 0 else "강아지")

plt.figure(figsize=(2, 2))
plt.imshow(test_scaled[check_index].reshape(64, 64), vmax=1, vmin=0, cmap="gray")
plt.axis("off")
plt.show()