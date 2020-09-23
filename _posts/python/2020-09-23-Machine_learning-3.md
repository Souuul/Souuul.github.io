---
title:  "[Python] Machine Learning-3"
header:
  teaser: "/assets/image/1200px-Pandas_logo.svg.png"
categories: 
  - Python
tags:
  - Machin learning
  - Scikit-learn
---

## Simple Linear Regression

이번 시간에는 실제 예제를 가지고 Linear Regression을 이용하여 예측 값을 산출해보도록 하겠습니다.

예제는 하기표처럼 온도와 오존량에 대한 데이터를 Training data로 사용할 예정이며 파일은 추가적으로 제공하도록 하겠습니다.

|      | Ozone | Solar.R | Wind | Temp | Month |  Day |
| ---: | ----: | ------: | ---: | ---: | ----: | ---: |
|    0 |  41.0 |   190.0 |  7.4 |   67 |     5 |    1 |
|    1 |  36.0 |   118.0 |  8.0 |   72 |     5 |    2 |
|    2 |  12.0 |   149.0 | 12.6 |   74 |     5 |    3 |
|    3 |  18.0 |   313.0 | 11.5 |   62 |     5 |    4 |
|    4 |   NaN |     NaN | 14.3 |   56 |     5 |    5 |
|  ... |   ... |     ... |  ... |  ... |   ... |  ... |
|  148 |  30.0 |   193.0 |  6.9 |   70 |     9 |   26 |
|  149 |   NaN |   145.0 | 13.2 |   77 |     9 |   27 |
|  150 |  14.0 |   191.0 | 14.3 |   75 |     9 |   28 |
|  151 |  18.0 |   131.0 |  8.0 |   76 |     9 |   29 |
|  152 |  20.0 |   223.0 | 11.5 |   68 |     9 |   30 |

저번시간과 똑같은 절차로 예측모델을 만들어 보도록 하겠습니다.

>1. Raw Data Loading
>2. Data Preprocessing ( 데이터 전처리 )
>3. Training Data Set
>4. 초기 W, b 세팅
>5. Loss function 정의
>6. 학습 예측 함수생성
>7. 기타 프로그램에사 필요한 변수 정의
>8. 학습진행
>9. 예측값 확인

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.my_library.machine_learning_library import numerical_derivative as nd

# 1. Raw Data Loading
df = pd.read_csv('./data/ozone.csv')
# display(df)

# 2. Data Preprocessing(데이터 전처리)
# - 결측치 처리...
# - 삭제, 값을 변경(평균, 최대, 최소), 값을 예측해서 값을 대체 
# - 이상치 처리(outlier)
# - 이상치를 검출하고, 변경하는 작업
# - 데이터 정규화 작업
# - 학습에 필요한 컬럼을 추출, 새로 생성.


# 필요한 column 만 추출
# 결치값을 제거!!
#

training_data = df[['Temp','Ozone', ]]
# display(training_data)
# print(training_data.shape) #(153, 2)


# 결측값을 제거할꺼에요.. 리스크가 있지만 그냥 이렇게 진행해요!!

training_data = training_data.dropna(how='any')
# display(training_data)
# print(training_data.shape) #(116, 2)

# 3. Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
t_data = training_data['Ozone'].values.reshape(-1,1)

# 4. 지금 우리는 Simple Linear Regression
#    y = Wx + b
# 그래서 우리가 구해야 하는 W,b를 정의
W = np.random.rand(1,1)
b = np.random.rand(1)

# 5. loss function 정의
def loss_func(x, t):
    y = np.dot(x, W) + b
    return np.mean(np.power(t-y,2)) # 최소제곱법

# 6. 학습종료 후 
def predict(x):
    return np.dot(x,W) +b

# 7. 기타 프로그램에서 필요한 변수들을 정의
learning_rate = 1e-4

f = lambda x: loss_func(x_data, t_data)

# 8. 학습을 진행!
for step in range(30000):
    W -= learning_rate * nd(f, W)
    b -= learning_rate * nd(f, b)
    
    if step %3000 ==0:
        print ('W : {}, b : {}, loss : {}'.format(W, b, loss_func(x_data,t_data)))
    
# 9. 그래프로 확인해 보아요!
plt.scatter(x_data, t_data)
plt.plot(x_data, np.dot(x_data, W)+b, 'r')
plt.show()

print(predict(62))
```

