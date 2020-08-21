---
title:  "[Git Hub] 가상환경"
header:
  teaser: "assets/image/Git.png"
categories: 
  - Git
tags:
- Git
- Github
- Python
---

# 파이썬 가상 환경

## 사용법

* 가상환경 생성

  * <code>venv</code> 라는 이름의 가상환경을 생성

    ``` python
    python3 -m venv {가상환경이름}
    python3 -m venv venv 
    ```

  * 가상환경을 생성하면 해당 디렉토리에 <code>venv</code> 폴터가 생성된다.

* 가상환경 실행

  ``` python
  $ source venv/Scripts/activate			# git bash용
  $ source venv/Scripts/activate.bat	# cmd용
  $ source venv/Scripts/activate.psl	# 파워쉘 용
  (venv) $
  ```

* 가상환경을 실행시킨 상태에서 파이썬 패키지(pip)를 설치하게 되면 venv 폴더의 Lib폴더에 설치를 하게 된다.

  * 해당프로젝트를 위한 패키지들을 따로 관리할 수 있다. 


## pip

```
# requirements.txt에 설치된 패키지 기록
$ pip freeze > requirements.txt
# requirements.txt에 설치된 패키지들을 설치
$ pip install -r requirements.txt
```

  