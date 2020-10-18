---
title:  "[Python] Deep Learning - Perceptron"
header:
  teaser: "/assets/image/1200px-TensorFlowLogo.svg.png"
categories: 
  - Python
tags:
  - Deep learning
---
## Perceptron 

Perceptron 이란 개념은 인간의 뇌처럼 사고하는 인공신경망입니다. 하지만 초기에는 XOR이라는 간단한 문제도 풀수없는 것에 한계를 가졌습니다. 

하지만 MultiLayer Perceptron, MLP 개념 즉 입력층과 출력층 사이에 은닉층을 생성하여 해결할 수 있었습니다. 

<p align ='center'><img src="../../assets/image/1*-IPQlOd46dlsutIbUq1Zcw.png" alt="Multi layer Perceptron (MLP) Models on Real World Banking Data | by Awhan  Mohanty | Becoming Human: Artificial Intelligence Magazine" style="zoom:30%;" /></p>

하기 예제를 보며 개념을 코드로 구현해 보도록 하겠습니다.

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report 


# Training Data Set
x_data = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]], dtype = np.float32)

t_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

# placeholder
X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
T = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# weight & bias // hidden layer
W2 = tf.Variable(tf.random.normal([2,100]), name='weight2')
b2 = tf.Variable(tf.random.normal([100]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(X, W2) + b2)

W3 = tf.Variable(tf.random.normal([100,6]), name='weight3')
b3 = tf.Variable(tf.random.normal([6]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random.normal([6,1]), name='weight4')
b4 = tf.Variable(tf.random.normal([1]), name='bias4')

# Hypothesis
logit = tf.matmul(layer3, W4) + b4
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit, labels=T))

# train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

# session 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, loss_val = sess.run([train, loss], feed_dict={X:x_data, T:t_data }) # trian과 loss를 둘다 실행
    
    if step %3000 == 0:
        print('loss : {}'.format(loss_val))
        
# 성능평가 (Accuracy)        
# print(classification_report('정답', '예측값'))
accuracy = tf.cast(H >=0.5, dtype=tf.float32)
result = sess.run(accuracy, feed_dict={X:x_data})
print(classification_report(t_data.ravel(), result.ravel()))

```

