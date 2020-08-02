---
title:  "[Django] Poll 프로젝트 1"
header:
  teaser: "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
categories: 
  - Django
tags:
  - Polls
  - Project
  - Webapplication


---
<h2>Polls 프로젝트 1</h2>
<h3>Templates 생성</h3>
저번시간에는 기본뼈대까지 생성하였으며 투표를 하기위한 질문과 선택지를 데이터베이스에 저장해보았습니다.

이번 시간에는 데이터베이스에 있는 데이터를 활용해보도록 하겠습니다.

투표 프로젝트를 만들면서 자세하게 알아보겠습니다.

일단 터미널에서 polls폴더 안에 template 폴더를 생성합니다.

><code>$ cd polls </code> 
>
><code>$ mkdir template </code>

Pycharm을 다시한번 열어보겠습니다. 

만들어진 template 폴더안에 index.html 파일을 생성합니다. 

생성하는 이유는 차근차근 설명하도록 하겠습니다.

어제 만든 models.py 안에 저희가 Class Question 과 Choice 를 만들었죠?

오늘은 어제 만든 Class를 사용하여 만들어보도록 하겠습니다.

polls폴더안에 views.py파일을 실행시켜 하기 코드를 추가합니다.

``` python

def index(request): # 서버가 보내준 request 반드시 인자로 넘겨줘야함 !!
    #로직처리 코드가 나와요!!
    tmp = Question.objects.all().order_by('-pub_date')[:3]
    # 객체(objects)의 모든것(all())을 불러옴 정렬 order_by() - 오름차순 [:] slicing
    context = {"latest_question_list" : tmp}
    return render(request, 'index.html', context)     #render는 그리는 작업 / HTML을 그림

```

<h4>코드설명</h4> 

---

서버와 클라이언트 사이에는 request와 response로 데이터를 주고 받습니다.

index(request) : 서버에서 받은 request를 index 함수에 인자로 넘겨줍니다. 

`tmp`라는 변수를 설정하여 

`Question` : Question Class

`objects` : 클래스 안의 객체

`all()` : 전부

`order_by('-pub_date')` : pub-date 기준으로 오름차순으로

`[:3]` : Slicing 3개까지 변수로 할당합니다.

dictionary 형태를 통하여 변수 context를 선언합니다. 

latest_question_list는 Key값  Question 의 모든객체는 Value 값이 됩니다.

`render`라는 것은 HTML을 변경하는 거라고 보시면 되겠습니다.

---

다음은 urls.py 파일을 생성해보겠습니다. 

처음부터 새로 만들어도 가능하지만 mysite에서 만들어진 urls.py를 polls하위에 복사하여 사용하도록하겠습니다.

모든내용을 지우고 

``` python

from django.urls import path
from . import views

app_name = "polls"

urlpatterns = [
    # http://localhost:8000/polls/
    path('', views.index, name='index'),

]

```

<h4>코드설명</h4>

---

from . import views 경로안에 view.py 함수를 사용합니다.

path('', views.index, name='index'),
localhost:8000/polls/ 경로뒤에 '' 아무것도 오지않으면 view.index 함수호출해요
name 은 경로에 대한 이름 이라고 보시면 되며 이것은 다음에 설명하도록 하겠습니다.



---


다시 처음으로 돌아가서 index.html의 코드를 수정하겠습니다.

<img src="/assets/image/Djangoindeximage.png" alt="Djangoindeximage">


<h4>코드설명</h4>

---

template code!! python도아니고 HTML 도아닌 template 안에서만 사용가능 
중괄호 표현은 template code이며 HTML, Python 언어가 아니므로 주의하셔서 사용해야합니다.



>```{ % % }``` : 로직코드이며 조건 및 반복문 등 로직을 사용해야할 경우 사용합니다.
>
>```{ { } }``` : 값을 입력할 경우 사용합니다.
>
>```{ { if latest_question_list} }``` 
>
> views.index에서 키값으로 받은 latest_question_list 존재여부를 판단합니다.
><br><br>
>
>```{ { for question in latest_question_list} }``` 
>
> question 변수에 lastest_question_list객체를 삽입합니다.
><br><br>
>
> ```<li><a href="/polls/{ { question.id } }">{ { question.question_text } }</a><li>```
>
> lastest_question_list 에서 전달받은 객체에서 .question_text 객체를 뽑아냅니다. 
><br><br><br>
>이렇게하면 초기페이지를 완성할 수 있습니다.