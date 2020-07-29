---
title:  "Python_Formatting"
header:
  teaser: "/assets/image/Formatting.png"
categories: 
  - Python
tags:
  - Text sequence
  - Data Type
  - 문자열
  - 데이터타입
  - str
  - formatting

---
<h2>In Not In 연산자</h2>
오늘은 `파이썬`의 `문자열` 의 `Formatting` 에 대하여 알아보도록 하겠습니다.

Formatting은 변수의 값을 원하는 곳에 입력하거나 출력이 가능합니다. 

예제를 통하여 다시 한번 정확하게 알아보겠습니다.

``` python
num_of_apple = 10
myStr = "나는 사과를 %d개 가지고 있어요!" % num_of_apple #// %d (숫자)
myStr1 = "나는 사과를 {}, 바나나 {}개 가지고 있어요!" .format(num_of_apple, 20)
myStr2 = "나는 사과를 {1}, 바나나 {0}개 가지고 있어요!" .format(num_of_apple, 20)
print(myStr) #나는 사과를 10개 가지고 있어요!
print(myStr1) #나는 사과를 10, 바나나 20개 가지고 있어요!
print(myStr2) # 나는 사과를 20, 바나나 10개 가지고 있어요!
```

복잡한 문장도 formatting 을 이용하여 원하는 위치에 변수의 값을 입력 및 출력을 할 수 있습니다.

오늘은 파이썬의 `formatting` 에 대하여 알아보았습니다.