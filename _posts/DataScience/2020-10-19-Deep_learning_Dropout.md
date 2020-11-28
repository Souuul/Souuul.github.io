---
title:  "[Data Science] Deep Learning - Dropout(TF_1.15)"
header:
  teaser: "/assets/image/1200px-TensorFlowLogo.svg.png"
categories: 
  - DataScience
tags:
  - Deep learning
  - Tensorflow
---
## Dropout -TF 1.15

`Dropout` 이란 overfitting을 막기위한 방법입니다. 데이터에서 어느정도의 비율을 제외하고 학습을 시키는 방법입니다. 하지만 `Dropout`의 경우 정해진 비율안에 랜덤으로 제외시키고 제외된 항목이 계속 변경되어 좀더 효율적이게 학습시키는 방법이라고 보시면 되겠습니다.

핵심코드(1.15 버전)는 하기와 같습니다. 

```python 
tf.nn.dropout(드롭아웃할레이어, rate = dropout_rate)
```

#### MNIST_Dropout 예제

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.model_selection import train_test_split # train, test 데이터분리
from sklearn.model_selection import KFold # cross validation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

tf.reset_default_graph()

df = pd.read_csv('/Users/admin/Downloads/Digit_Recognizer_train.csv')

display(df)
# 결측치 확인
print(df.isnull().sum())
x_data = df.iloc[:,1:]
t_data = df['label']

# Data split
x_data_train, x_data_test, t_data_train, t_data_test=\
train_test_split(x_data,t_data,test_size = 0.3, random_state = 0)

# 데이터 정규화 (Normalization)
x_scaler = MinMaxScaler()
x_scaler.fit(x_data_train)
x_data_train_norm = x_scaler.transform(x_data_train)
x_data_test_norm = x_scaler.transform(x_data_test)

sess = tf.Session()   # Tensorflow node를 실행하기 위해서 session을 생성

# One-hot encoding 
t_data_train_onehot = sess.run(tf.one_hot(t_data_train, depth=10))  
t_data_test_onehot = sess.run(tf.one_hot(t_data_test, depth=10))


# Placeholder
X = tf.placeholder(shape = [None, 784], dtype = tf.float32)
T = tf.placeholder(shape = [None, 10], dtype = tf.float32)
dropout_rate = tf.placeholder(dtype = tf.float32)

# Weight & bias
W2 = tf.get_variable('weight2',shape=[784,256],
                     initializer = tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([256]), name='bias2')
_layer2 = tf.nn.relu(tf.matmul(X, W2) + b2)
layer2 = tf.nn.dropout(_layer2, rate = dropout_rate)


W3 = tf.get_variable('weight3',shape=[256,128], 
                     initializer = tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([128]), name='bias3')
_layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
layer3 = tf.nn.dropout(_layer3, rate=dropout_rate)


W4 = tf.get_variable('weight4',shape=[128,10], 
                     initializer = tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([10]), name='bias4')

# Hypothesis
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)     # Multinomial Hypothesis

# Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                                 labels=T))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)



num_of_epoch = 100
batch_size = 100

# 7. 학습진행
def run_train(sess,train_x, train_t):
    print('### 학습 시작 ###')
    sess.run(tf.global_variables_initializer())  # tf.Variable 초기화(W,b)

    total_batch = int(train_x.shape[0] / batch_size)
    for step in range(num_of_epoch):

        for i in range(total_batch):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_t = train_t[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([train,loss], feed_dict={X:batch_x,
                                                            T:batch_t,
                                                           dropout_rate:0.3})
        if step % 10 == 0:
            print('Loss : {}'.format(loss_val))
    print('### 학습 끝 ###')

    
    
# Accuracy
predict = tf.argmax(H,1)   # [[0.1 0.3  0.2 0.2 ... 0.1]]

# sklearn을 이용해서 classification_report를 출력해보아요!!

# train데이터로 학습하고 train데이터로 성능평가를 해 보아요!!  

run_train(sess,x_data_train_norm,t_data_train_onehot)

target_name = ['num 0', 'num 1', 'num 2', 'num 3',
               'num 4', 'num 5', 'num 6', 'num 7',
               'num 8', 'num 9']


print(classification_report(t_data_test,
                            sess.run(predict, feed_dict={X:x_data_test_norm,
                                                        dropout_rate:0}),
                            target_names=target_name))
```

