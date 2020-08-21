---
title:  "[Kakao API] Refresh token 발급받기"
header:
  teaser: "assets/image/kakao.png"
categories: 
  - API
tags:
- Kakao
- API
- Token
---
<h2>Kakao API [Refresh token]</h2>

오늘은 `Refresh token`에 대하여 알아보겠습니다. 

아직 사용자 토큰을 받지 못하신 분은 [[Kakao API] 사용자토큰](/api/kakao_api_usertoken/) 을 참고하여 사용자토큰을 먼저 받고 오시면 되겠습니다. Refresh token 토큰은 사용자 토큰을 받는방법과 비슷합니다.



<h3>Refresh token 받기</h3>

여러분들이 직접 카카오에 있는 사용법을 보고 사용할 수 있도록 설명하겠습니다.

먼저 토큰을 발급받기 위해 [카카오 개발자 사이트](https://developers.kakao.com/)에 접속하여 회원가입합니다.

<h3>사용자 토큰 받기</h3>

![kakao_token](../../assets/image/kakao_token.png)



<h4>응답값</h4>

##### Key

| Name                     | Type      | Description                                                  |
| :----------------------- | :-------- | :----------------------------------------------------------- |
| token_type               | `String`  | 토큰 타입, "bearer"로 고정                                   |
| access_token             | `String`  | 사용자 액세스 토큰 값                                        |
| expires_in               | `Integer` | 액세스 토큰 만료 시간(초)                                    |
| refresh_token            | `String`  | 사용자 리프레시 토큰 값                                      |
| refresh_token_expires_in | `Integer` | 리프레시 토큰 만료 시간(초)                                  |
| scope                    | `String`  | 인증된 사용자의 정보 조회 권한 범위 범위가 여러 개일 경우, 공백으로 구분 |



사용자 토큰을 받기위해 코드를 상기 URL 에서 필수요소들을 request하고 응답값을 .json 형식의 파일로 저장해보겠습니다. 



```python
import requests
import json
#
# 초기 키 땡기기
app_key = {REST_API_KEY} # 초기 앱키 rest_key
code = "JrFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

url = "https://kauth.kakao.com/oauth/token"
data = {
     "grant_type"    : "authorization_code",
     "client_id"     : app_key,
     "redirect_url"  : "https://localhost.com",
     "code"          : code
}

response = requests.post(url, data=data)

tokens = response.json()

print(tokens)

with open("kakao_token.json", 'w') as fp:
    json.dump(tokens, fp) # 저장하는 것
```



`response = requests.post(url, data=data)` 코드를 통하여 post방식으로 요청하면 응답 값을 받을 수 있습니다.



``` python
with open("kakao_token.json", 'w') as fp:
    json.dump(tokens, fp)
```

상기 코드를 통하여 응답값을 `kakao_token.json` 이름의 .json 형태의 파일로 저장합니다.

이렇게 하여 저희는 사용자토큰을 kakao_token.json 파일에 넣어보았습니다. 사용자 토큰을 바로 사용하는 것도 가능하지만 .json 형태의 파일로 저장하여 구분하였습니다.



오늘은 카카오 API 사용자 토큰을 받아보았습니다.

