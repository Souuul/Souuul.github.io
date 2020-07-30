var store = [{
        "title": "[Python] Idexing, Slicing (str Data Type)",
        "excerpt":"Indexing &amp; Slicing  오늘은 파이썬의 문자열 의 Idexing 과 Slicing을 알아보겠습니다.   Indexing #문자열에 번호에 해당하는 문자를 추출하려면??   python은 배열이 존재 하지 않아요! 하지만 다른언어에서는 - index 사용가능합니다.   그렇다면 파이썬의 문자열에서 번호 혹은 순서에 맞는 문자를 추출하기 위해서는 어떻게 해야할까요 ?   하기 그림은 문자열에 해당되는 번호를 명시한 그림입니다.    \t   그렇다면 그림에 해당하는 번호를 파이썬 코드로 구현해볼까요?   my_var = 'HELLO' print(my_var[1])  #E   Slicing #문자열에 범위를 선정하여 추출하려면??   Slicing은 말 그대로 잘라내다입니다. Indexing과의 차이점은 번호가 아닌 범위로 문자열을 추출이 가능합니다.   하기 예제를 확인하면서 익혀보도록 하겠습니다.   my_var = 'HELLO' print(my_var[0:3]) #HEL print(my_var[0:]) #HELLO print(my_var[0:-1]) #HELL print(my_var[-2:]) #LO print(my_var[:]) # HELLO    오늘은 파이썬의 Indexing과 Slicing에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","Idexing","Slicing"],
        "url": "http://localhost:4000/python/Idexing-Slicing/",
        "teaser": "https://miro.medium.com/max/2646/1*GokwxHxq5I-Myy3_ummrtw.png"
      },{
        "title": "[Python] In Not in (str Data Type)",
        "excerpt":"In Not In 연산자  오늘은 파이썬의 문자열 의 in 과 not in 연산자에 대하여 키워드를 통하여 알아보도록 하겠습니다.   In, Not in 연산자 #문자나 문자열이 포함되어 있는지 확인하려면 ??   예제를 통하여 다시 한번 정확하게 알아보겠습니다.   myStr = \"This is a sample Test\" print(\"sample\" in myStr) # True / 대소문자 구별 print(\"sample\" not in myStr) #False / 대소문자 구별   즉 sample이라는 문자열이 myStr이라는 변수에 포함되어있는지 안되어 있는지 확인이 가능합니다.   오늘은 파이썬의 in과 not in에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","In","Not in"],
        "url": "http://localhost:4000/python/In-Not/",
        "teaser": "http://localhost:4000/assets/image/INNOTIN.png"
      },{
        "title": "[Python] Numeric Data Type",
        "excerpt":"Numeric Data Type (숫자형)  오늘은 파이썬의 숫자형 데이터 타입에 대하여 알아보겠습니다.   데이터 숫자는 크게 3가지로 나눌 수가 있습니다!      int(정수)   float(실수)   complex(복소수)   하기 예제를 보면서 파이썬의 특징을 알아보겠습니다.  a = 100 # 정수 b = 3.14159265358979 # 실수 c = 1+ 2j # 복소수 d = 0o34 # 8 진수 e = 0xAB # 16 진수  파이썬이 아닌 다른 프로그램언어를 배우신 분이라면 변수선언을 할때 var같은 선언을 하지않고 간편하게 변수를 선언을 할 수 있습니다.   Type # 데이터 타입에 대하여 알기를 원한다면??  print (type(a)) # class int print (type(b)) # class float print (type(c)) # class complex print (type(d)) # class int print (type(e)) # class int   Python의 연산의 특징에 대하여 알아보아요!  my_result = 3 / 4 # 0이 아니라 0.75  프로그래밍에서는 버림으로 표시 print(my_result) # 0.75   my_result = 10 % 3 #나머지 연산자 print(my_result) # 1  my_result = 10 // 3 #몫 연산자 print(my_result) # 3   오늘은 파이썬의 숫자형 데이터 타입에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Numeric","Data Type","숫자형","데이터타입"],
        "url": "http://localhost:4000/python/Numeric-Data-Type/",
        "teaser": "http://localhost:4000/assets/image/Numeric.png"
      },{
        "title": "Python Page",
        "excerpt":" \t   파이썬 페이지   숫자형 데이터 타입   문자열   문자열 연산   In Not in   문자열 Indexing &amp; Slicing   Formatting   List  ","categories": ["Python"],
        "tags": [],
        "url": "http://localhost:4000/python/PythonPage/",
        "teaser": "http://localhost:4000/assets/image/python.jpg"
      },{
        "title": "[Python] Text operation (str Data Type)",
        "excerpt":"문자열의 연산 (문자열, str)  오늘은 파이썬의 문자열 연산에 대하여 알아보겠습니다.   일상생활에서 말을 할때 대부분 단일언어로 대화를 하지않습니다. 문자를 조합하거나 합성해서 문장을 만들어 대화를 합니다.   파이썬의 문자열도 마찬가지로 합치거나 원하는 문자열을 추가를 할 수가 있습니다. 자세한 내용은 하기 예제를 통하여 알아보겠습니다.   first = 'haha' second = 'hoho'  print(first + second) #hahahoho print(first + str(10)) # error 자동으로 숫자를 문자로 변경 X ( Java 에서는 가능 합니다 !) print(first*3) #hahahahahaha   상기의 예제는 문자열을 더하거나 곱하여서 새로운 문자열을 생성하는 것을 보았습니다.   앞으로 프로젝트를 진행하면서 받는 데이터들이 완벽한 문자열 혹은 문자로 받아진다면 저러한 연산은 필요없겠지만 저희가 앞으로 받은 데이터는 예상할 수 없는 데이터이기 때문에 항상 가공을 해야합니다.   문자열의 연산은 앞으로의 프로젝트에서 데이터를 처리하기위한 ‘첫걸음’ 정도로 생각하면 되겠습니다.   오늘은 파이썬의 문자열 연산에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","operation"],
        "url": "http://localhost:4000/python/Text-operation/",
        "teaser": "http://localhost:4000/assets/image/Text%20Operation.png"
      },{
        "title": "[Python] Text sequence (str Data Type)",
        "excerpt":"Text sequence (문자열, str)  오늘은 파이썬의 문자열 데이터 타입에 대하여 알아보겠습니다.   다른 언어는 문자와 문자열을 구분합니다. 문자는 한글자 예를들어 a 같이 단일 문자로 이뤄진 것을 의미합니다.   문자열은 두글자 이상으로 이루어진 문자열을 의미하며 apple, graph같은 단어를 의미합니다.   다른언어에서는 문자를 표현할때 ' ', 문자열을 표현할때는 \" \" 으로 표현합니다.   파이썬에서는 어떨까요? 하기 예제를 보면서 다른언어와 파이썬의 차이를 살펴보도록 하겠습니다.  a = 'Hello' b = \"K\" c = 'python'  print(a)  # Hello print(b)  # K print(c)  # python   상기 예제를 확인해보면 파이썬에서는 문자와 문자열을 표현할때 ' ' 와 \" \" 을 구분하지않고 사용이 가능합니다.   오늘은 파이썬의 문자열에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str"],
        "url": "http://localhost:4000/python/Text-sequence/",
        "teaser": "http://localhost:4000/assets/image/Text%20Squence.png"
      },{
        "title": "테크블로그에 오신것을 환영합니다.",
        "excerpt":" \t   한솔   1990.04.09   경력 사항    육군 중위 (2013.03 - 2015.06)    아트라스콥코 (2016.11 - 2020.06)    KSA (OREAN STANDARDS ASSOCIATION) 전문요원    기업출강 및 제휴강의 담당 (삼성전자, SK 하이닉스, 현대자동차, LG전자 등)    외국계 기업 B2B 마케팅 담당    공장 개선 컨설팅 및 스마트팩토리 컨설팅 담당    각종 고객 세미나, 전시회 및 테크 쇼 기획 및 실행 담당    마케팅 캠페인 기획 및 관리 담당    비디오, 리플렛, 카탈로그 등 마케팅 자료 기획 및 제작 담당    기업 및 산업용 공구 사업부 SNS채널 및 뉴스레터 플랫폼 기획 및 관리 담당    KC인증, Cleanroom certification, 안전인증 담당    기술 특허 담당   보유기술    2D, 3D 설계 / 3D Printing    Co-bot operation 및 로봇 제어    PLC    Python    HTML, CSS, JavaScript  ","categories": ["Welcome"],
        "tags": [],
        "url": "http://localhost:4000/welcome/welcome/",
        "teaser": "https://cdn.pixabay.com/photo/2015/05/09/23/46/welcome-sign-760358__480.jpg"
      },{
        "title": "[Python] Formatting",
        "excerpt":"In Not In 연산자  오늘은 파이썬의 문자열 의 Formatting 에 대하여 알아보도록 하겠습니다.   Formatting은 변수의 값을 원하는 곳에 입력하거나 출력이 가능합니다.   예제를 통하여 다시 한번 정확하게 알아보겠습니다.   num_of_apple = 10 myStr = \"나는 사과를 %d개 가지고 있어요!\" % num_of_apple #// %d (숫자) myStr1 = \"나는 사과를 {}, 바나나 {}개 가지고 있어요!\" .format(num_of_apple, 20) myStr2 = \"나는 사과를 {1}, 바나나 {0}개 가지고 있어요!\" .format(num_of_apple, 20) print(myStr) #나는 사과를 10개 가지고 있어요! print(myStr1) #나는 사과를 10, 바나나 20개 가지고 있어요! print(myStr2) # 나는 사과를 20, 바나나 10개 가지고 있어요!   복잡한 문장도 formatting 을 이용하여 원하는 위치에 변수의 값을 입력 및 출력을 할 수 있습니다.   오늘은 파이썬의 formatting 에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","formatting"],
        "url": "http://localhost:4000/python/Formatting/",
        "teaser": "http://localhost:4000/assets/image/Formatting.png"
      },{
        "title": "[Python] List",
        "excerpt":"List  오늘은 파이썬의 List 에 대하여 알아보도록 하겠습니다.   List는 임의의 객체(데이터)를 순서대로 저장하는 집합 자료형입니다.   List는 literal로 표현할 떄 [ ] 대괄호로 표현합니다.   List는 어떻게 표하는지 아래의 코드를 보며 알아보겠습니다.   my_list = [] print(type(my_list))  # &lt;class ‘list’&gt; my_list = list()      # 함수를 이용하여 리스트를 제작 my_list = [1, 2, 3]   # ~ : code convention _ 가독성이 좋게 표현하게 Hint 기능 제공 my_list = [1, 2, 3.14, \"Hello\", [5, 6, 7], 100] # 중첩리스트, 2차원이 아님    List는 문자열과 마찬가지로 Indexing 과 Slicing 모두 가능합니다. 물론 연산도 가능합니다.   Indexing과 Slicing의 개념이 궁금하신분들은 문자열 Indexing &amp; Slicing 편을 참고하시기 바랍니다.   간단한 예제를 통해서 자세하게 알아보겠습니다.   List의 Indexing 과 Slicing   print(my_list[1]) #2 print(my_list[-2]) #[5, 6, 7] print(my_list[4:5]) #[5, 6, 7] // list의 Slicing 은 list print(my_list[-2][1]) #6 print(my_list[0:2]) #[1, 2]   List의 연산  a = [1, 2, 3] b = [4, 5, 6] print (a + b) # [1, 2, 3, 4, 5, 6] list 의 합은 하나의 리스트로 생성 # 단 행렬에서의 연산은 [5,7,9] numpy에서 사용시 주의 할 것 print (a*3) # [1, 2, 3, 1, 2, 3, 1, 2, 3]  a = [1, 2, 3] a[0] = 5 print(a) # [5, 2, 3]  a[0] = [7, 8, 9] print(a) # [[7, 8, 9], 2, 3]  a[0:1] = [7, 8, 9] print(a) # [7, 8, 9, 2, 3]    list 값 추가 및 변경   List의 경우 값의 추가 및 변경이 가능합니다   # append   a = [1, 2, 3] a.append(4) 끝에 추가하는 것 print(a) #[1, 2, 3, 4] a.append([5, 6, 7]) print(a) #[1, 2, 3, [5, 6, 7]]  # sort   my_list = [\"홍길동\", \"아이유\", \"강감찬\", \"신사임당\", \"Kim\"] result = my_list.sort() # 리스트를 오름차순으로 정렬_ 1 2 3 4 5 print(result) # None    오늘은 파이썬의 List 에 대하여 알아보았습니다.  ","categories": ["Python"],
        "tags": ["Sequence Type","Data Type","List"],
        "url": "http://localhost:4000/python/List/",
        "teaser": "http://localhost:4000/assets/image/List.png"
      },{
        "title": "[Django] 이론",
        "excerpt":"Django  오늘은 Django에 대하여 알아보아요! Python 으로 만들어진 오픈소스 웹 어플리케이션을 쉽게 작성할 수 있도록 도와주는 Framework   라이브러리(library)  특수한 처리를 하기 위해서 만들어 놓은 함수집합이 Library입니다. jQuery도 라이브러리라고 볼 수 있어요!   장점 : 내가 모든걸 다 작성할 필요가 없어요!   단점(특징) : 전체 프로그램의 로직을 담당하지는 않아요!   예) jQuery 를 이용해서 영화정보를 출력하는 문제를 구현할 때 사람마다 구현이 제각각…   프레임 워크  프로그램의 전체적인 로직부분이 이미 구현이 되어 있어요!   그래서 프레임워크를 사용할때는 기본적으로 사용되는 코드가 제공되요! (스케폴딩- scaffolding)   유지보수성이 좋아집니다. 단, 처음에 프레임워크의 동작원리를 이해하는 작업이 필요!!      Django를 이용하면 Web Application에서 많이 자주 구현해야 하는 내용을 쉽게 구현할 수 있어요!   라이브러리(library) VS 프레임워크  두 가지 모두 이미 만들어진 것을 사용하는 것은 동일합니다. 차이점은 규칙에 있습니다. 라이브러리는 가져다 쓰는 개념이고 프레임워크는 내 것을 가져다가 프레임 워크 규칙에 맞추는 개념입니다.   Django의 특징  Django는 MVC Model을 기반으로 한 MVT model을 이용해요!   MVC Model : Model, View, Controller   MVT model : Model, View, Template   model : 데이터베이스 처리   View : 로직을 담당   Template : 클라이언트에게 보여줄 화면을 담당   오늘은 파이썬의 Django의 이론에 대하여 알아보았습니다.  ","categories": ["Django"],
        "tags": [],
        "url": "http://localhost:4000/django/Django-Basic/",
        "teaser": "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
      },{
        "title": "[Django] 설치 및 기초운용",
        "excerpt":"Django설치 및 기초운용  1. Django 설치  mac 환경은 설치하기가 조금어려워서 가상환경으로 진행하겠습니다. 윈도우 사용자분들은 python 경로(path)가 제대로 설정되어있는지 확인해주세요.   터미널에서 pip를 이용해서 Djanggo를 설치합니다.   pip(python install package), pypl(python Package Index)라는 repository 에 있는 Django를 설치합니다.   가상환경(virtualenv) 설치 $ python3 -m venv posts   가상환경 실행(virtualenv) $ source posts/bin/activate   설치된 패키지 확인 $ pip3 freeze   장고설치 '$ pip3 install django   python 3.7.8버전을 사용하시는 분은 pip3를 update를 해주셔야 합니다. $ pip3 install --upgrade pip   여기까지 하시면 설치가 완료가 됩니다.    2. Project Setup   시작은 프로젝트의 뼈대를 만드는 일부터 시작 폴터를 하나 만들어요!! 폴더를 하나만들어요   mkdir python-Django cd python-Django django-damin startproject mysite   Scaffolding 하자!  cd python-Django django-admin startproject mysite   python-Django/myself 이름을 MyFirstWebPoll로 바꿈 (폴더 안에 있는 프로젝트 설정을 담고 있는 mysite폴더와 헤깔리니까) 프로젝트 안에 개별 어플리케이션을 집어넣자 cd MyFirstWebPoll python3 manage.py startapp polls   poll 이라는 어플이 생성되고 필요한 파일들이 scaffolding 됨   이제는 pycharm 으로 처리하자   PyCharm 처리과정 MyFirstWebPoll프로젝트 실행 &gt; setting.py를 이용해 프로젝트 설정처리 기본테이블(기본DB)을 생성 장고는 관리자 모드 화면을 제공 관리자 ID PW가 DB어딘가에는 저장이 되어 있어야함 (DB설정이 전제됨) python3 manage.py migrate   서버기동을 해보아요   내장된 테스트 서버를 이용해서 우리 프로젝트를 deploy해 보아요 python3 manage.py runserver # 서버실행 python3 manage.py createsuperuser #관리자 설정 python3 manage.py runserver # 서버실행      Polls application 구현 1. Model 생성부터 해야 해요!!  Database : 데이터의 집합체  DBMS (Database Management System) :데이터베이스를 구축하고 원하는 정보를 추출하고 새로운 데이터를 입력하고 기존데이터를 삭제하고 기존데이터를 수정하는 작업을 진행. (프로그램) sqlite3라는 DBMS를 Django에서는 default 로 사용   이런 데이터베이스는 언제부터 사용됐을까?? 아주아주 초창기에는 데이터를 이렇게 관리하면 좋지않을까 라고 생각했습니다.   계층형 데이터베이스   순번\t이름\t학과 1\t홍길동\t심리학과 2\t김길동\t컴퓨터학과 3\t신사임당\t경제학과 …\t…\t…   이렇게 사용하다가… 시간이 지나서…  한사람이 등장해서 논문을 발표.. Relation이라는 주제로 발표   IBM에서 논문내용이 좋으니.. 우리 이걸가지고 DBMS를 만들어보아요!! DB2 성공! 성능도 좋고 사용하기도 편하고 다른모든 회사들도 이논문을 근간으로 DBMS를 만들어서 판매   Relational Database(관계형 데이터베이스)   객체지향의 개념이 들어가서.. 결국 객체관계형 데이터베이스가 탄생   테이블 자료를 끌어가야하는데 프로그램 방식은 두가지가 있어요!     ODBC            ORM (Object Relation Mapping) # Django           Table = relation = class와 매핑되며   Class Student: stu_num\t-순번 stu_num\t-이름 stu_num\t-학과   ㅁ\t1 ㅁ\t홍길동 ㅁ\t심리학과   Python 으로 relation 을 만들어 봅시다!!     Model 작업은 우리가 사용하는 Database에 Table을 생성하는 작업이에요! (Table == Relation)   model.py 를 수정해보아요 ! class 만들기   model에서제작   admin 에서 등록   from polls.models import Question, Choice  Register your models here.   #괄호안에는 내가 등록할 클래스가 나와야해요! admin.site.register(Question) admin.site.register(Choice)   여기까지는 구조를 아직만든게 아니에요!!   표를 삽입을 해줘야해요   python3 manage.py makemigrations python3 manage.py migrate python3 manage.py runserver     ","categories": ["Django"],
        "tags": ["Install","Webapplication"],
        "url": "http://localhost:4000/django/Django-simple-web-application/",
        "teaser": "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
      },{
        "title": "Django Page",
        "excerpt":" \t   파이썬 페이지   Djanggo 기초이론   Djanggo 설치 및 운용  ","categories": ["Django"],
        "tags": [],
        "url": "http://localhost:4000/django/DjangoPage/",
        "teaser": "https://live.staticflickr.com/3595/3475465970_7044242629_b.jpg"
      }]
