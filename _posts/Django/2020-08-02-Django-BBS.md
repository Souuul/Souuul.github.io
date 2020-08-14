---
title:  "[Django] BBS Project "
header:
  teaser: "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
categories: 
  - Django
tags:
  - Polls
  - Project
  - Webapplication
  - BBS


---

<h2> BBS (Bullentin Board System) </h2>
Poll 프로젝트에 이어 오늘은 게시판을 만들어 보겠습니다.

Poll 프로젝트는 처음 부터 끝까지 저희가 코드를 작성했었죠 ?

외울 것도 많고 작성해야하는 것도 많았습니다.

오늘은 polls project 에서 배웠던 내용을 기반으로 쉽게 제작해보겠습니다.

ModelForm을 이용해서 CRUD구현을 알아 보겠습니다.

<h3>CRUD (CREATE READ UPDATE DELETE)</h3>

ModelForm을 이용하면 사용자 입력양식 처리하는게 쉬워집니다. 

여기에 html 프로젝트에서 사용하였던 Bootstrap도 포함해서 자동으로 

만들어진 component를 통해 Web application을 만들어 보겠습니다. 


1. 필요한 package들을 설치해야해요!
 
 기본적으로 Django를 설치해야합니다. 
 > <code>$ pip3 install Django</code>

 추가적으로 bootstrap에 대한 package를 설치합니다.

일반적인 HTML 파일을 만들고 Bootstrap을 CDN과 tag 속성을 이용하면 Bootstrap을 이용할 수 있어요!

그런데 이번에는 ModelForm을 이용할 것이고 사용자 입력양식 HTML을 자동으로 만들어줘요 

자동으로 생성되기 떄문에 Bootstrap을 적용할수 없어요

이런경우에 사용자 입력양식에 Bootstrap을 적용하기 위해서 특정 Package를 설치해야 해요!

> <code>$ pip3 install Django-bootstrap4</code>


2. project를 생성 + application 생성

Django는 framework 이고 당연히 scaffolding 기능을 제공합니다.

특정 명령어를 이용해서 필요한 폴더와 파일을 자동으로 생성해야 합니다.

터미널을 이용해서 working directory를 python-django 폴더로 변경해요!!

<code>$ django-admin startproject blog</code>

해당 명령을 실행하면 기본적인 프로젝트 구조가 만들어져요

우리 프로젝트와 앞으로 생성할 application 을 포함하고 있는 폴더의 이름을 변경합니다

파일명이 혼동여부가 있어 상위폴더이름을 MyBlogSystem으로 파일명을 변경합니다.

하나의 application을 우리프로젝트에 추가해요!!

<code>python3 manage.py startapp posts</code>


3. project 설정 (settings)
app등록 
bootstrap도 마찬가지로 application 등록을 해야함.

bootstrap4 등록!!

settings.py 에 APP을 추가합니다.

 ```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'posts.apps.PostsConfig',
    'bootstrap4'
]
```

Templates 에서 DIRS를 추가 경로지정가능 !
```python
        'DIRS': [os.path.join(BASE_DIR, 'blog', 'templates')],
```


맨아래 내려가서 하기코드도 추가합니다. static이라는 폴더에 바로접근할 수 있습니다.
정적리소스를 사용가능합니다.

```python
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static')
]
```

4. 우리의 project를 web에 deploy해봐야 해요 !!
기본적인 데이터 베이스를 들고가야해서 무조건 사용자 등록을 해야합니다. 

admin page가 존재 web에 deploy하기 이전에.. 기본 table부터 생성해야해요!
> ```python3 manage.py migrate```

관리자 계정이 있어야 Admin page(관리자 화면)를 사용할 수 있어요!
> ```python3 manage.py createsuperuser```

설정을 다했으니 이제 Web에 deploy해보도록 하겠습니다. 
> ```python3 manage.py runserver```

----------------

프로젝트의 기본설정이 완성 되었습니다.

5. 모델구현
기능을 구현하러 가야해요!! application을 구현해야해요!
기능을 구현할 때 제일먼저해야 하는 일은 
사용할 데이터에 대한 정확한 명세를 작성하는 거에요!!

Django는 ORM을 이용하기 떄문에 class를 이용해서 Database를 사용해요!


Model을 만들어야해요!!
posts / applicaiton / models.py 파일에 Model을 정의

CharField vs TextField 
한줄 vs 여러줄 

이렇게 내가 만든 model을 Admin page에 반영하기 위해서

admin.py에 class를 등록해야해요!!

Model을 생성했기 떄문에 데이터베이스에 변경이 필요!

데이터베이스를 이렇게 저렇게 변경하세요 라는 명세(초안) 가 필요!!
> python3 manage.py makemigrations

초안이 완성되면 실제로 데이터베이스에 적용해서 Table을 생성
> python3 manage.py migrate

초기화 하기위해서는 migrations 폴더에 initial.py 와 db.sqlite3를 전부다 지우면됨!

6. URL 경로 설정

from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from django.views.generic.base import TemplateView

htttp:// localhost:8000 요청이 들어왔을때 우리 전체 project의 홈페이지로 이동할거에요!!
Django는 elegant URL을 지원해요!!
정규표현식 (regualr expression)
시작 > ^, 끝 > $
[0-9] : 1글자를 지칭
{} : 반복횟수 {3} 3번반복 {3,5} 3 아님 5번 반복
[0-9]{4} : 4자리 숫자가 나옴
r(raw) 은 escape 문자를 한번 더 사용하지 않도록 처리.
r"^[0-9]{1,3}$" 숫자가 1개나 3개나 모두가능
\d 숫자를 지칭
r"^010[1-9]\d{6,7}$"


urlpatterns = [
    # view 를 거치지 않고 바로 html을 호출
    url(r"^$", TemplateView.as_view(template_name='index.html'), name="home"),
    path('admin/', admin.site.urls),
    # path('posts/', include('posts/urls'))
]


cdn 방식으로 bootstrap을 써봐요 !!
cover의 소스를 끌고와요 F12 개발자모드에서 소스끌어오기!
static /css 에 붙여넣기

url 설정 복붙하고 app_name = posts'

urlpatterns = [
    path('list/', views.p_list, name = 'list'),
]

blog project 안에 urls.py 부터 설정

post application 안에 url.py 설정

base template html 파일ㅇ르 blog project 안에 templates 폴더안에 생성



7. ModelForm 생성
사용자 입력양식을 우리가 직접 HTML template안에 입력하는게 아니라 Model을 기반으로 사용자 입력양식을 자동으로 생성해 줄 수 있는데 ModelForm을 이용하면 이 작업을 할 수 있어요 !!

class를 작성해야 해요 ! > 어느파일에서 만들어야 하나요??

forms.py 에서 정의해야해요!

from django import forms
from posts.models import Post

class PostForm (forms.ModelForm):
    class Meta:
        model = Post
        fields = ['author', 'contents']

8. list page 생성
- views.py 을 수정해서 list/ 가 요청되었을 때 해야하는 일을 기술


from django.shortcuts import render
from posts.models import Post
# Create your views here.

def p_list(request):    #데이터
    my_list = Post.objects.all().order_by('-id')
    context = {'posts' : my_list}
    return render(request, 'list.html', context)


posts/ list.html

전부 다 작성할필요가없어요 base.html의 것을 확장해서 사용

{ % extends 'base.html' % }

{ % block container % }
    <div class ='container'>
        <h1>Bullentin Board System(BBS)</h1>

    </div>
{ % endblock % }

안에 좀더 내용을 채워봐요 !!


9. create page 생성

10. delete 기능 구현