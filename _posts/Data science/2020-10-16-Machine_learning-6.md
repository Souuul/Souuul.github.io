---
title:  "[Python] Evaluation - Tensorflow"
header:
  teaser: "/assets/image/1200px-TensorFlowLogo.svg.png"
categories: 
  - Python
tags:
  - Machin learning
  - Sklearn
---
## Evaluation - Tensorflow 

평가를 동일하게 Tensorflow 를 통해서 진행해보겠습니다.

```python
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv('./data/bmi.csv', skiprows=3)

# 결측치 확인
df.isnull().sum()

# 이상치 처리 (제거)
zscore_threshold = 1.8

for col in df.columns:
    outlier = df[col][np.abs(stats.zscore(df[col]))>zscore_threshold]
    df = df.loc[~df[col].isin(outlier)]

# data split
x_data_train, x_data_test, t_data_train, t_data_test =\
train_test_split(df[['height', 'weight']], df['label'], test_size=0.3, random_state=0)

# placeholder
X = tf.placeholder(shape = [None, 2],dtype=tf.float32)
T = tf.placeholder(shape = [None, 3],dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random.normal([2,3]), name='weight')
b = tf.Variable(tf.random.normal([3]), name='bias')

# Hypothesis
logit = tf.matmul(X,W) + b
loss =  tf.nn.softmax(logit)

# Train
sess = tf.Session()
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

num_of_epoch = 1000
batch_size = 100

def run_train(sess, x_data_train, y_data_train):
    print ('학습시작 !')
    sess.run(tf.global_variables_initializer())
    total_batch = int (num_of_epoch / batch_size)
    for step in range(num_of_epoch):
        for i in range(total_batch):
            batch_x = x_data_train[i*batch_size: (i+1)batch_size]
            batch_t = t_data_train[i*batch_size: (i+1)batch_size]
            
            _, loss_val = sess.run([train, loss], feed_dict={X=batch_x, T= batch_t})
        if step % 100 == 0:
            print('loss : {}'.format(loss_val))
    print('학습종료')

# Accuracy 
predict = tf.argmax(T,1)
correct = tf.equal(predict, tf.argmax(T,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

run_train(sess, x_data_train_norm, t_data_train_onehot)

# Accuracy 측정 (Training data 로 validation을 수행해보아요!)
result = sess.run(accuracy, feed_dict={X:x_data_train_norm,T:t_data_train_onehot})
# Training data 로 validation한 정확도 : 0.9827142953872681

```

### Cross Validation

``` python
# Cross Validation 
cv = 5 # [훈련, 검증] => 5 Set가 만들어져요
results = [] # 5 set 에 대한 accuracy를 구해서 ㅣist 안에 차곡
kf = KFold(n_splits=cv, shuffle = True)


for training_idx, validation_idx in kf.split(x_data_train_norm):
    print(training_idx, validation_idx)
    train_x = x_data_train_norm[training_idx] # Fancy indexing
    train_t = t_data_train_onehot[training_idx]
    valid_x = x_data_train_norm[validation_idx]
    valid_t = t_data_train_onehot[validation_idx]
    
    run_train(sess, train_x, train_t)
    results.append(sess.run(accuracy, feed_dict={X:valid_x, T:valid_t}))

print('cross Validation 결과 : {}'.format(results))
print('cross Validation 최종결과 : {}'.format(np.mean(results)))

'''
cross Validation 결과 : [0.98321426, 0.9810714, 0.9810714, 0.9771429, 0.9867857]
cross Validation 최종결과 : 0.9818571209907532
'''
```

### Prediction

```python
height = 187
weight = 78

my_state = [[height, weight]]
my_state_scaled = scaler.transform(my_state)
print(my_state_scaled)

result = sess.run(H, feed_dict={X:my_state_scaled})
print(result)
print(np.argmax(result))
'''
[[0.8375     0.95555556]]
[[5.762022e-04 9.758552e-01 2.356866e-02]]
1
'''
```

