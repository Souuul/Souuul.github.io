---
title:  "[Django] 설치 및 기초운용"
header:
  teaser: "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
categories: 
  - Django
tags:
  - Install
  - Webapplication


---
<h2>Django설치 및 기초운용</h2>
<h3>Django 설치</h3>
mac 기반으로 작성하였으니 참고하시기 바랍니다. python 실행경로를 꼭! 확인해 주세요.

`pip`(python install package), `pypi`(python Package Index)라는 repository에 있는 Django를 설치합니다.

<h4>가상환경에서 설치</h4>
<u>가상환경</u>에서 하실분은 하기 내용을 추가하여 설치해주시기 바랍니다.

>1. 가상환경(virtualenv) 설치 : `$ python3 -m venv posts`
>
>2. 가상환경 실행(virtualenv) : `$ source posts/bin/activate`
>
>3. 설치된 패키지 확인 : `$ pip3 freeze`


<h4>장고설치</h4>
<u>가상환경이 아닌 로컬</u>에서 설치하실 분들은 이쪽부터 진행 하시면 됩니다.

>1. Django 설치 : `'$ pip3 install django`
>
>2. python 3.7.8버전을 사용하시는 분은 pip3를 update를 해주셔야 합니다.
>
>3. pip upgrade : `$ pip3 install --upgrade pip`
>
>4. 여기까지 하시면 설치가 완료가 됩니다.


<h3> Project Setup </h3>
본격적으로 프로젝트의 뼈대를 만드는 일부터 시작하겠습니다. 

<h4> 터미널에서 설정 </h4>
>1. 폴더 생성 : `$ mkdir python-Django`
>
>2. 만들어진 경로로 이동 : `$ cd python-Django`
>
>3. mysite 프로젝트 생성 및 Scaffolding : `$ django-damin startproject mysite`
>
>4. 폴더명 변경(선택사항) :python-Django/myself 이름을 MyFirstWebPoll로 변경
>
>5. 변경된 경로로 이동 : `$ cd MyFirstWebPoll`
>
>6. 프로젝트 안에 개별 어플리케이션을 생성 :`$ python3 manage.py startapp polls`
>
>7. poll 이라는 어플이 생성되고 필요한 파일들이 scaffolding 됨

<h4> Python 설정변경 [pycharm 사용] </h4>

> 1. pycharm에서 MyFirstWebPoll 프로젝트 실행
>
> 2. setting.py를 이용해 프로젝트 설정처리
>
> 3. 기본테이블(기본DB)을 생성
>
> 4. 장고는 관리자 모드 화면을 제공
>
> 5. 관리자 ID PW가 DB어딘가에는 저장이 되어 있어야함 (DB설정이 전제됨)
>
> 6. INSTALLED_APPS 리스트 항목에 'polls.apps.PollsConfig' 를 추가
>
> 7. TIME_ZONE 'Asia/Seoul' 로 변경

```python
# Application definition 

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'polls.apps.PollsConfig'
]
```
...
```python
# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_L10N = True

USE_TZ = True
```


<h4> 프로젝트 Deploy</h4>
>1. 터미널을 이용해서 내장 서버를 이용하여 프로젝트를 `deploy` 해보도록 하겠습니다.
>
>2. 프로젝트 migrate : `python3 manage.py migrate`
>
>3. 서버 실행 : `python3 manage.py runserver`
>
>4. 관리자 설정 : `python3 manage.py createsuperuser` #user, email, pw 설정
>
>5. 서버실행 : `python3 manage.py runserver` 
>
>6. <code>127.0.0.1:8000 로 접속
>
>7. Polls application 구현확인


---

<h4> 데이터 베이스 설명 </h4>
Database : 데이터의 집합체

DBMS (Database Management System)

데이터베이스를 구축하고 원하는 정보를 추출하고 새로운 데이터를 입력하고 기존데이터를 삭제하고 기존데이터를 수정하는 작업을 진행.


Django에서는 sqlite3라는 DBMS를 default 로 사용합니다. 

이런 데이터베이스는 언제부터 사용됐을까요?
초창기에는 데이터를 이렇게 관리하면 좋지않을까 라고 생각했습니다.

|계층형 데이터베이스|

|||| 순번 |||| 이름     |||| 학과       ||
|||| ---- |||| -------- |||| ---------- ||
|||| 1    |||| 홍길동   |||| 심리학과   ||
|||| 2    |||| 김길동   |||| 컴퓨터학과 ||
|||| 3    |||| 신사임당 |||| 경제학과   ||
|||| …    |||| …        |||| …        ||


IBM에서 Relation 이라는 논문을 발표하였고 DB2를 출시하였습니다.

현재에는 거의 모든 DBMS가 Relational Database(관계형 데이터베이스)으로 제작됩니다.

결국 객체관계형 데이터베이스가 탄생하게 되었습니다.

테이블 자료를 끌어가야하는데 프로그램 방식은 크게 두가지 방식이 있습니다.
1. ODBC
2. ORM (Object Relation Mapping) # Django 

즉 쉽게 설명하자면 Table = relation = class와 매핑됩니다.




---




<h4> Model 생성 </h4>
Model 작업은 우리가 사용하는 Database에 Table을 생성하는 작업이에요!
(Table == Relation)

>models.py 파일 열기


```python
from django.db import models

# Create your models here.
class Question(models.Model): # Django 가제공하는 models 클래스
    question_text = models.CharField(max_length=200)    #문자열을 받고 길이는 200자 까지
    pub_date = models.DateTimeField('date published')

    def __str__(self):      # 일반적으로 연산을 위해서 문자열로 변환을 하고 싶을때!
        return self.question_text

class Choice(models.Model):
    # question_id = models.ForeignKey(Question, on_delete=models.CASCADE)
    # 기본적으로 _id가 자동으로 붙음
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    # 만약에 Question 에 대하여 지운다면 같이 지운다 CASCADE 문제없이 만드는 것

    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text




    # def __repr__(self):     # class 의 객체를 문자열로 표현할 경우에 사용

    # Table의 id column은 default로 지정되요!!!
    # id가 primary key로 설정이 가능!, Not null (값이 무조건 들어가는 경우), Unique (겹치는 값이 안옴)
    # id는 autoincrement 특성(값이 들어오면 자동적으로 증가하는 특성)을 가지는 정수형으로 지정
    # 자동으로 생성해주기 때문에 class정의에서 나오지 않아요!!

```



>/polls/admin.py

``` python
from polls.models import Question, Choice
# Register your models here.

#괄호안에는 내가 등록할 클래스가 나와야해요!
admin.site.register(Question)
admin.site.register(Choice)
```

여기까지는 구조를 아직만든게 아니에요!! 

하기 과정을 통하여 표를 삽입을 해줘야합니다. 

><code>python3 manage.py makemigrations
>
><code>python3 manage.py migrate
>
><code>python3 manage.py runserver
>
><code>127.0.0.1:8000 로 접속

완료하면 하기 그림처럼 초기 설정 화면은 얻을수 있습니다. 
<img src="/assets/image/Django project1.png" alt="Django project1" style="width:500px">





