---
title:  "[Algorithm] Greedy Algorithm"
header:
  teaser: "assets/image/Algorithm.png"
categories: 
  - Algorithm
tags:
- Greedy

---

## Greedy Algorithm

오늘은 `Greedy Algorithm`에 대하여 알아보겠습니다. 

Greedy Algorithm은 말 그대로 탐욕 알고리즘입니다. 탐욕 알고리즘은 최적해를 구하는데 사용되는 근사적인 방법으로 최적이라고 생각되는 것을 선택해 나가는 방식으로 정답에 도달하는 방법입니다. 전체적인 상황을 고려하지 않고 현재상황에서 가장 최적의 것을 선택하는 방법입니다. 

모든문제를 순차적으로 풀면 정확한 정답을 얻을 수 있겠으나 시간적으로나 용량적으로 효율적이지 못한경우에는 알고리즘의 다양한 방법을 통하여 좀더 효율적이게 문제를 해결할 수 있습니다.

그럼 바로 `Greedy Algorithm`에 대하여  몇가지 문제를 통하여 알아보도록 하겠습니다. 

#### 거스름돈 문제 

 물건값은 n으로 주어집니다. 거스름돈의 동전의 종류는 4종류 이며 500원 100원 50원 10원이며 거스름돈의 동전 개수를 최대한 적게 하려고 할때의 동전개수를 출력하는 프로그램을 작성해보세요.

``` python
n = 800
coin = [500, 100, 50, 10]
count = 0
for i in coin:
  #큰 동전부터 먼저 거슬러줍니다.
  count += n//i
  # 동전을 거슬러주고 남은돈이 다음동전에서 연산되도록 처리합니다.
  n = n%i
print(count)	#4
```

화폐의 종류가 K 라고 할때 소스코드의 시간복잡도는 *O(K)* 입니다. 



#### 연산 문제 

어떠한 수 N이 1이 될 때까지 다음 두가지 연산 중 한가지만 수행하려고합니다. 

1. N 에서 1을 뺍니다.

2. N을 K로 나눕니다. (단 N이 K로 나눠질 경우에만 사용 가능)

1이 될때까지의 연산의 횟수를 출력하는 프로그램을 작성해보세요. 

예) N이 17 이고 K 가 4 이면 정답은 3입니다.

``` python
N = 17
K = 4
count = 0
while N > 1:
  if N % K == 0:
    N = N/K
  else:
    N = N-1
  count += 1

 print (count) #3
```

좀더 효율적으로 풀어보도록 하겠습니다.

``` python
N = 17
K = 4
count = 0
while True:
  # N을 K로 나눈 몫을 K로 다시곱해서 -1을 해야하는 부분을 한번에 추출함
  target = (N//K)*K
  count += n - target
  
  if N < K:
    break
  N //= K
  count += 1

# 마지막으로 남은 수에 대하여 카운트를 해줌
count += (N-1)
print(count)
```



#### 곱하기 혹은 더하기

각 자리가 숫자 (0~9)로만 이루어진 문자열 S가 주어졌을때 왼쪽부터 오른쪽까지 하나씩 모든숫자를 확인하면 +혹은 X연산자를 넣어 결과적으로 만들어질 수 있는 가장 큰 수를 구하는 프로그램을 작성하세요.

예) 02984 = (0+2)X9X8X4 = 576
``` python
S = input()

sum_num = int(S[0])

for i in range(1, len(S)):
  if int(S[i]) <=1:
    sum_num += int(S[i])
  else:
    sum_num *= int(S[i])
    
print(sum_num)
```

좀더 효율적으로 풀어보도록 하겠습니다.

``` python
S = input()
sum_num = 0
for i in range(0, len(S)):
    sum_num = max(sum_num + int(S[i]), sum_num * int(S[i]))
print(sum_num)
```

오늘은  `Greedy Algorithm`에 대하여 알아보았습니다.

