---
title:  "[Django] 이론"
header:
  teaser: "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
categories: 
  - Django
tags:
#  - Sequence Type


---
<h2>Django</h2>
오늘은 `Django`에 대하여 알아보아요!
Python 으로 만들어진 오픈소스
웹 어플리케이션을 쉽게 작성할 수 있도록 도와주는 Framework

<h3>라이브러리(library)</h3>
특수한 처리를 하기 위해서 만들어 놓은 함수집합이 `Library`입니다. jQuery도 라이브러리라고 볼 수 있어요!

장점 : 내가 모든걸 다 작성할 필요가 없어요!

단점(특징) : 전체 프로그램의 로직을 담당하지는 않아요!

예) jQuery 를 이용해서 영화정보를 출력하는 문제를 구현할 때 사람마다 구현이 제각각...

<h3>프레임 워크</h3>
프로그램의 전체적인 로직부분이 이미 구현이 되어 있어요!

그래서 프레임워크를 사용할때는 기본적으로 사용되는 코드가 제공되요! (스케폴딩- scaffolding)

유지보수성이 좋아집니다. 단, 처음에 프레임워크의 동작원리를 이해하는 작업이 필요!!

- Django를 이용하면 Web Application에서 많이 자주 구현해야 하는 내용을 쉽게 구현할 수 있어요!

<h3>라이브러리(library) VS 프레임워크</h3>
두 가지 모두 이미 만들어진 것을 사용하는 것은 동일합니다. 차이점은 규칙에 있습니다. 라이브러리는 가져다 쓰는 개념이고 프레임워크는 내 것을 가져다가 프레임 워크 규칙에 맞추는 개념입니다.


<h3> 데이터 베이스 </h3>
Database : 데이터의 집합체

DBMS (Database Management System)

데이터베이스를 구축하고 원하는 정보를 추출하고 새로운 데이터를 입력하고 기존데이터를 삭제하고 기존데이터를 수정하는 작업을 진행.


Django에서는 sqlite3라는 DBMS를 default 로 사용합니다. 

이런 데이터베이스는 언제부터 사용됐을까요?
초창기에는 데이터를 이렇게 관리하면 좋지않을까라고 생각했습니다.
그래서 나온 것이 조직도 형상 트리 구조로 정의된 계층형 데이터 베이스입니다.


하지만 IBM에서 Relation 이라는 논문을 발표하였고 DB2를 출시하였습니다.

관계형 데이터베이스는 행과 열로 구성된 테이블 간의 관계를 나타내는 것입니다. 

|관계형 데이터베이스|

|||| 순번 |||| 이름     |||| 학과       ||
|||| ---- |||| -------- |||| ---------- ||
|||| 1    |||| 홍길동   |||| 심리학과   ||
|||| 2    |||| 김길동   |||| 컴퓨터학과 ||
|||| 3    |||| 신사임당 |||| 경제학과   ||
|||| …    |||| …        |||| …        ||


데이터를 관리하기 편한 장점을 가지고 있어 현재에는 거의 모든 DBMS가 Relational Database(관계형 데이터베이스)으로 제작됩니다.

테이블 자료를 끌어가야하는데 프로그램 방식은 크게 두가지 방식이 있습니다.
1. ODBC
2. ORM (Object Relation Mapping) # Django 

즉 쉽게 설명하자면 Table = relation = class와 매핑됩니다.




---


<h3>Django의 특징</h3>
`Django`는 MVC Model을 기반으로 한 MVT model을 이용해요!

MVC Model : Model, View, Controller 

MVT model : Model, View, Template

model : 데이터베이스 처리

View : 로직을 담당

Template : 클라이언트에게 보여줄 화면을 담당





오늘은 파이썬의 `Django`의 이론에 대하여 알아보았습니다.