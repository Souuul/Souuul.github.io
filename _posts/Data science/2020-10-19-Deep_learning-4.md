---
title:  "[Python] Deep Learning - Dropout(TF_2.0)"
header:
  teaser: "/assets/image/1200px-TensorFlowLogo.svg.png"
categories: 
  - Python
  - Tensorflow
tags:
  - Deep learning
---
## Dropout - TF 2.0

`Dropout` 이란 overfitting을 막기위한 방법입니다. 데이터에서 어느정도의 비율을 제외하고 학습을 시키는 방법입니다. 하지만 `Dropout`의 경우 정해진 비율안에 랜덤으로 제외시키고 제외된 항목이 계속 변경되어 좀더 효율적이게 학습시키는 방법이라고 보시면 되겠습니다.

핵심코드(2.0 버전)는 하기와 같습니다. 

```python 
keras_model.add(Dense(420, activation='relu', kernel_initializer='he_uniform'))
keras_model.add(Dropout(0.3))
```

#### MNIST_Dropout 예제

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder # Normalization
from sklearn.model_selection import train_test_split # train, test 데이터분리
from sklearn.model_selection import KFold # cross validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from scipy import stats
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report


df = pd.read_csv('/Users/admin/Downloads/Digit_Recognizer_train.csv')
display(df)
# 결측치확인
print(df.isnull().sum())


# Data split
x_data_train, x_data_test, t_data_train, t_data_test =\
train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size = 0.3, random_state=0)

# Normalization
x_data_scaler = MinMaxScaler()
x_data_scaler.fit(x_data_train)
x_data_train_norm = x_data_scaler.transform(x_data_train)
x_data_test_norm = x_data_scaler.transform(x_data_test)
t_data_train_onehot = to_categorical(t_data_train)
t_data_test_onehot = to_categorical(t_data_test)


# TF 2.0 구현 
keras_model = Sequential()
keras_model.add(Flatten(input_shape=(784,))) # 아래행처럼 더하기가 가능
keras_model.add(Dense(420, activation='relu', kernel_initializer='he_uniform'))
keras_model.add(Dropout(0.3))
keras_model.add(Dense(258, activation='relu', kernel_initializer='he_uniform'))
keras_model.add(Dropout(0.3))
keras_model.add(Dense(10, activation='softmax', kernel_initializer='he_uniform'))


keras_model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

''' t_data를 onehot encoding 하지 않을경우 하기 코드를 사용하셔도 무방합니다. 
keras_model.compile(optimizer=Adam(learning_rate=1e-3),
                    loss='sparse_categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'])
'''

history = keras_model.fit(x_data_train_norm, t_data_train_onehot, 
                         epochs = 100, 
                          verbose = 1, 
                          batch_size=128, 
                          validation_split=0.3)

predict_val = np.argmax(keras_model.predict(x_data_test_norm), axis=1)
```

``` python
import matplotlib.pyplot as plt
print(classification_report(t_data_test, predict_val.ravel()))

print(type(history))
print(history.history.keys())
plt.plot(history.history['val_categorical_accuracy'], color='b')
plt.plot(history.history['categorical_accuracy'], color='r')
plt.show()
```

