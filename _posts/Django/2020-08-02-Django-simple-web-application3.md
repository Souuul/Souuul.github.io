---
title:  "[Django] Poll 프로젝트 2"
header:
  teaser: "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
categories: 
  - Django
tags:
  - Polls
  - Project
  - Webapplication


---
<h2>Polls 프로젝트 2</h2>
<h3>투표시스템 질문지 항목 추가</h3>
저번시간에는 Question 항목까지 만드는 것을 해보았습니다.

이번 시간에는 선택항목까지 제작해보도록 하겠습니다.

Pycharm을 다시한번 열어보겠습니다. 

만들어진 template 폴더안에 detail.html 파일을 생성합니다. 

생성하는 이유는 차근차근 설명하도록 하겠습니다.

첫 시간에 models.py 안에 저희가 Class Question 과 Choice 를 만들었죠?

오늘도 만든 Class를 사용하여 만들어보도록 하겠습니다.

polls폴더안에 views.py파일을 실행시켜 하기 코드를 추가합니다.

``` python

def detail(request, question_id):
    # 로직처리를 해요!
    # 아까는 모든 Question 객체를 다 구해서 리스트로 만들었는데
    # 이번에는 특정 Question 객체 1개만 구해야 해요
    tmp = get_object_or_404(Question, pk=question_id)
    context = {"question" : tmp} # questiond 이라는 문자열로 context를 호출합니다.
    return render(request, 'detail.html', context)

```

<h4>코드설명</h4> 

---

index method를 제작했을 때와는 다르게 처음보는 코드가 있습니다.

>`def detail(request, question_id):`
>
> `def index` 에서 와는다르게 question_id 라는 인자를 하나 더받습니다.
> 
> index에서 하이퍼링크를 타고 detail 페이지로 넘어올때 그 인자까지도 받는다는 내용입니다.
>
>`get_object_or_404(Question, pk=question_id)`
>
> 받는 객체가 있다면 객체를 받고 없다면 404 page를 띄운다는 간단한 내용입니다. 
>
> 대신 전달받은 primary key에 한해 Question에서의 객체를 받는 다는 내용입니다. 

---

다음은 urls.py 파일을 생성해보겠습니다. 

>http://localhost:8000/polls/ 주소로 접속하면 index.html 로 들어가게 되어있습니다.
>
>저희는 질문지를 눌렀을때 선택항목이 나오는 페이지가 나오게 경로를 잡아줘야합니다. 
>
>다양한 방법이 있지만 index에서 전달받은 pk를 가지고 접속해보도록 하겠습니다.
>
하기코드를 입력하겠습니다. 


``` python
    # http://localhost:8000/polls/<숫자>/
    path('<int:question_id>/', views.detail, name='detail')

]


```

<h4>코드설명</h4>

---
'path('<int:question_id>/', views.detail, name='detail')'

> 'http://localhost:8000/polls/question_id '
>
> polls/question_id 의 경로일 경우 view.index 함수를 실행합니다.

---


다시 처음으로 돌아가서 detail.html의 코드를 수정하겠습니다.

``` html
<body>
    <h1>{ {question.question_text} }</h1>

    <form action="" method="post">
    { % csrf_token % }
    { % for choice in question.choice_set.all % }
        <input type="radio" id="choice{ {forloop.counter} }"
               name="choice"
               value="choice.id"
        >
        <rabel for="choice{ {forloop.counter} }">
            { {choice.choice_text} }
        </rabel>
        <br>
    { % endfor % }

        <input type="submit" value="투표">
    </form>

</body>

]

```


<h4>코드설명</h4>

---
`<form action="" method="post">`
> post 방식으로 데이터를 받으면 action으로 처리한다는 내용입니다. 
>
> submit 버튼을 눌렀을 경우 하기 input에서 name과 value 인자를 다음페이지로 보낼 수 있습니다.

>```{ { question.choice_set.all }}```
> 
> choice에 해당되는 객체를 모두 받습니다.


``` html
        <input type="radio" id="choice{{forloop.counter}}"
               name="choice"
               value="choice.id"
        >
```
> 라디오 버튼을 제작합니다. 
>
> `id` 는 choice1, ...
> 
> `name = choice` : 항목을 서로 묶기 위해서 명시합니다. 중복선택을 방지가능합니다.

``` html
        <rabel for="choice{ {forloop.counter} }">
            { {choice.choice_text} }
        </rabel>
```
> 상기 항목에서 명시한 id와 rabel 의 text와 매칭되어 관리 됩니다.
>
> `$ python3 manage.py runserver` 를 통해 수정된 페이지를 확인해봅시다.

<img src = '/assets/image/django_polls_detail.png'>

<br><br><br>
오늘은 poll 프로젝트의 질문지 선택까지 제작해보았습니다.