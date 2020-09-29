---
title:  "[Python] Machine Learning-4"
header:
  teaser: "/assets/image/1200px-TensorFlowLogo.svg.png"
categories: 
  - Python
tags:
  - Machin learning
  - Tensorflow
---

## Tensorflow 

이번시간에는 `Tensorflow`를 이용해서 Linear Regression을 해보도록 하겠습니다.

먼저 Tensorflow를 설치하고 시작해보도록 하겠습니다. Tensorflow는 버전이 1.xx 버전과 2.xx 버전이 있습니다. 이번시간에는 1.15버전을 설치하여 사용하도록 하겠습니다.

``` powershell
$ pip install tensorflow==1.15
```

설치를 하였으니 'Hello World'를 출력해보도록 하겠습니다.

```python
import tensorflow as tf
print(tf.__version__)

node = tf.constant('Hello World')

# 우리가 만든 그래프를 실행하기 위해서 Session이 필요!
sess = tf.Session()

# runner 인 Session 이 생성되었으니 이걸 이용해서 node를 실행해 보아요!
print(sess.run(node)) # b'Hello World' b는 바이트
print(sess.run(node).decode())  #Hello World'
```

이번에는 덧셈을 수행해보도록 하겠습니다.

``` python
# placeholder를 이용
# 2개의 수를 입력으로 받아서 덧셈연산을 수행
import tensorflow as tf
node1 = tf.placeholder(dtype = tf.float32) # scalar 형태의 값 1개를 실수로 받아들일 수 있는 Placeholder
node2 = tf.placeholder(dtype = tf.float32) # scalar 형태의 값 1개를 실수로 받아들일 수 있는 Placeholder
node3 = node1 + node2

sess = tf.Session()
sess.run(node3, feed_dict={node1:30,node2:20})
```

간단한 예제를 통해서 예측모델을 만들어보도록 하겠습니다.

``` python
import tensorflow as tf

# 1. Raw Data Loading
# 2. Data Preprocessing(데이터 전처리)
# 3. Training data set
x_data = [2,4,5,7,10]
t_data = [7,11,13,17,23]

# 4. Weight & bias 정의
W = tf.Variable(tf.random.normal([1]), name = 'weight') # W = np.random.rand(1,1)
b = tf.Variable(tf.random.normal([1]), name = 'bias') # b = np.random.rand(1)

# 5. hypothesis, simple Linear Regression Model
H = W * x_data + b

# 6. Loss function
loss = tf.reduce_mean(tf.square(t_data-H))
#7. train node 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = optimizer.minimize(loss)

# 8. 실행준비 및 초기화작업
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 초기화 작업

# 9. 반복해서 학습을 진행!!

for step in range(30000):
    _,W_val, b_val = sess.run([train,W,b])
    if step % 3000 == 0:
        print('W:{}, b:{}'.format(W_val, b_val))
        
print(sess.run(H)) # [ 6.9997516 10.999857  12.999908  17.000013  23.000172 ]

# 10. predict!!
print(sess.run(H, feed_dict={X:[6]})) # [13.111737]
```

