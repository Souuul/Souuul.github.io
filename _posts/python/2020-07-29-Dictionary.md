---
title:  "[Python] Dictionary"
header:
  teaser: "/assets/image/List.png"
categories: 
  - Python
tags:
  - Sequence Type
  - Data Type
  - Dictionary

---
<h2>Dictionary</h2>
오늘은 `파이썬`의 `Dictionary` 에 대하여 알아보도록 하겠습니다.

Dictionary Key 와 Value의 쌍으로 이루어진 자료형 데이터 타입입니다.

Dictionary literal로 표현할 떄 `{ }` 중괄호로 표현합니다.

Dictionary는 어떻게 표하는지 아래의 코드를 보며 알아보겠습니다.

``` python
my_dict = {}
print(type(my_dict))  # <class 'dict'>
my_dict = dict()      # 함수를 이용하여 딕셔너리를 제작
my_dict = {'a': '홍길동', 'b' : '강감찬'}
my_dict['c'] = '신사임당' # 키와 값의 추가 
print(my_dict) = {'a': '홍길동', 'b' : '강감찬', 'c' : '신사임당'}
```

몇가지 명령어를 통하여 Dictionary 구조에 대하여 한번 더 알아보겠습니다.

``` python
a = { "name" : "홍길동", "age" : 40, "address": "서울"}
print(a.keys()) # dict_keys(['name', 'age', 'address']) // 리스트는 아님
print(type(a.keys())) #<class 'dict_keys'>
print(list(a.keys())) # ['name', 'age', 'address']
print(a.values()) #dict_values(['홍길동', 40, '서울'])

print(a.items()) #dict_items([('name', '홍길동'), ('age', 40), ('address', '서울')]) 
# tuple로 표현 

# for문을 돌리기 위해서 튜플로 변환 ( dict의 경우에는 for 문을 돌릴 수 없음)

```

오늘은 파이썬의 `Dictionary` 에 대하여 알아보았습니다.