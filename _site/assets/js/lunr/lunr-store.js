var store = [{
        "title": "Python_Idexing, Slicing (str Data Type)",
        "excerpt":"오늘은 파이썬의 문자열 의 Idexing 과 Slicing을 알아보겠습니다.   Indexing #문자열에 번호에 해당하는 문자를 추출하려면??  python은 배열이 존재 하지 않아요! 하지만 다른언어에서는 - index 사용가능합니다. 그렇다면 파이썬의 문자열에서 번호 혹은 순서에 맞는 문자를 추출하기 위해서는 어떻게 해야할까요 ?   하기 그림은 문자열에 해당되는 번호를 명시한 그림입니다.    \t   그렇다면 그림에 해당하는 번호를 파이썬 코드로 구현해볼까요?   my_var = 'HELLO' print(my_var[1])  #E   Slicing #문자열에 범위를 선정하여 추출하려면?? Slicing은 말 그대로 잘라내다입니다. Indexing과의 차이점은 번호가 아닌 범위로 문자열을 추출이 가능합니다. 하기 예제를 확인하면서 익혀보도록 하겠습니다.   my_var = 'HELLO' print(my_var[0:3]) #HEL print(my_var[0:]) #HELLO print(my_var[0:-1]) #HELL print(my_var[-2:]) #LO print(my_var[:]) # HELLO    오늘은 파이썬의 Indexing과 Slicing에 대하여 알아보았습니다.  ","categories": ["Jekyll","Minimal mistake","Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","Idexing","Slicing"],
        "url": "http://localhost:4000/jekyll/minimal%20mistake/python/Idexing-Slicing/",
        "teaser": "https://miro.medium.com/max/2646/1*GokwxHxq5I-Myy3_ummrtw.png"
      },{
        "title": "Python_In Not in (str Data Type)",
        "excerpt":"오늘은 파이썬의 문자열 의 in 과 not in 연산자에 대하여 키워드를 통하여 알아보도록 하겠습니다.   In, Not in 연산자 #문자나 문자열이 포함되어 있는지 확인하려면 ??   예제를 통하여 다시 한번 정확하게 알아보겠습니다.   myStr = \"This is a sample Test\" print(\"sample\" in myStr) # True / 대소문자 구별 print(\"sample\" not in myStr) #False / 대소문자 구별   즉 sample이라는 문자열이 myStr이라는 변수에 포함되어있는지 안되어 있는지 확인이 가능합니다.   오늘은 파이썬의 in과 not in에 대하여 알아보았습니다.  ","categories": ["Jekyll","Minimal mistake","Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","In","Not in"],
        "url": "http://localhost:4000/jekyll/minimal%20mistake/python/In-Not/",
        "teaser": "http://localhost:4000/assets/image/INNOTIN.png"
      },{
        "title": "Python_Numeric Data Type",
        "excerpt":"Numeric Data Type (숫자형) 오늘은 파이썬의 숫자형 데이터 타입에 대하여 알아보겠습니다.  데이터 숫자는 크게 3가지로 나눌 수가 있습니다!     int(정수)   float(실수)   complex(복소수)   하기 예제를 보면서 파이썬의 특징을 알아보겠습니다.  a = 100 # 정수 b = 3.14159265358979 # 실수 c = 1+ 2j # 복소수 d = 0o34 # 8 진수 e = 0xAB # 16 진수  파이썬이 아닌 다른 프로그램언어를 배우신 분이라면 변수선언을 할때 var같은 선언을 하지않고 간편하게 변수를 선언을 할 수 있습니다.   Type # 데이터 타입에 대하여 알기를 원한다면??  print (type(a)) # class int print (type(b)) # class float print (type(c)) # class complex print (type(d)) # class int print (type(e)) # class int   Python의 연산의 특징에 대하여 알아보아요!  my_result = 3 / 4 # 0이 아니라 0.75  프로그래밍에서는 버림으로 표시 print(my_result) # 0.75   my_result = 10 % 3 #나머지 연산자 print(my_result) # 1  my_result = 10 // 3 #몫 연산자 print(my_result) # 3   오늘은 파이썬의 숫자형 데이터 타입에 대하여 알아보았습니다.  ","categories": ["Jekyll","Minimal mistake","Python"],
        "tags": ["Numeric","Data Type","숫자형","데이터타입"],
        "url": "http://localhost:4000/jekyll/minimal%20mistake/python/Numeric-Data-Type/",
        "teaser": "http://localhost:4000/assets/image/Numeric.png"
      },{
        "title": "Python_Text operation (str Data Type)",
        "excerpt":"문자열의 연산 (문자열, str) 오늘은 파이썬의 문자열 연산에 대하여 알아보겠습니다.  일상생활에서 말을 할때 대부분 단일언어로 대화를 하지않습니다. 문자를 조합하거나 합성해서 문장을 만들어 대화를 합니다. 파이썬의 문자열도 마찬가지로 합치거나 원하는 문자열을 추가를 할 수가 있습니다. 자세한 내용은 하기 예제를 통하여 알아보겠습니다.   first = 'haha' second = 'hoho'  print(first + second) #hahahoho print(first + str(10)) # error 자동으로 숫자를 문자로 변경 X ( Java 에서는 가능 합니다 !) print(first*3) #hahahahahaha   상기의 예제는 문자열을 더하거나 곱하여서 새로운 문자열을 생성하는 것을 보았습니다.   앞으로 프로젝트를 진행하면서 받는 데이터들이 완벽한 문자열 혹은 문자로 받아진다면 저러한 연산은 필요없겠지만 저희가 앞으로 받은 데이터는 예상할 수 없는 데이터이기 때문에 항상 가공을 해야합니다. 문자열의 연산은 앞으로의 프로젝트에서 데이터를 처리하기위한 ‘첫걸음’ 정도로 생각하면 되겠습니다.   오늘은 파이썬의 문자열 연산에 대하여 알아보았습니다.  ","categories": ["Jekyll","Minimal mistake","Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str","operation"],
        "url": "http://localhost:4000/jekyll/minimal%20mistake/python/Text-operation/",
        "teaser": "http://localhost:4000/assets/image/Text%20Operation.png"
      },{
        "title": "Python_Text sequence (str Data Type)",
        "excerpt":"Text sequence (문자열, str) 오늘은 파이썬의 문자열 데이터 타입에 대하여 알아보겠습니다.  다른 언어는 문자와 문자열을 구분합니다. 문자는 한글자 예를들어 a 같이 단일 문자로 이뤄진 것을 의미합니다. 문자열은 두글자 이상으로 이루어진 문자열을 의미하며 apple, graph같은 단어를 의미합니다.   다른언어에서는 문자를 표현할때 ' ', 문자열을 표현할때는 \" \" 으로 표현합니다. 파이썬에서는 어떨까요? 하기 예제를 보면서 다른언어와 파이썬의 차이를 살펴보도록 하겠습니다.  a = 'Hello' b = \"K\" c = 'python'  print(a)  # Hello print(b)  # K print(c)  # python   상기 예제를 확인해보면 파이썬에서는 문자와 문자열을 표현할때 ' ' 와 \" \" 을 구분하지않고 사용이 가능합니다.   오늘은 파이썬의 문자열에 대하여 알아보았습니다.  ","categories": ["Jekyll","Minimal mistake","Python"],
        "tags": ["Text sequence","Data Type","문자열","데이터타입","str"],
        "url": "http://localhost:4000/jekyll/minimal%20mistake/python/Text-sequence/",
        "teaser": "http://localhost:4000/assets/image/Text%20Squence.png"
      },{
        "title": "테크블로그에 오신것을 환영합니다.",
        "excerpt":"오늘은 파이썬에 대하여 알아보겠습니다.  Python 은 배우기 쉽습니다 !!!   이런글씨도 사용이 가능해요! 네이버로 이동!   _posts   Jekyll also offers powerful support for code snippets:  print(\"python 을 배워보아요!\")   print(\"한솔블로그에 오신것을 환영합니다. !!!\") def print_hi(name)   puts \"Hi, #{name}\" end print_hi('Tom') #=&gt; prints 'Hi, Tom' to STDOUT.   Check out the Jekyll docs for more info on how to get the most out of Jekyll. File all bugs/feature requests at Jekyll’s GitHub repo. If you have questions, you can ask them on Jekyll Talk.   ","categories": ["Jekyll","Python"],
        "tags": ["update","update","연습"],
        "url": "http://localhost:4000/jekyll/python/welcome/",
        "teaser": "https://cdn.pixabay.com/photo/2015/05/09/23/46/welcome-sign-760358__480.jpg"
      }]
