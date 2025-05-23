---
layout: single
title:  "[AI 부트캠프] 컴퓨터 공학 개론"
categories: Bootcamp
tag: [패스트캠퍼스, 패스트캠퍼스AI부트캠프, 업스테이지패스트캠퍼스, UpstageAILab, 국비지원, 패스트캠퍼스업스테이지에이아이랩, 패스트캠퍼스업스테이지부트캠프]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


***



이번 챕터느 어느 대학을 가던지 컴퓨터공학과라면 배우는 내용들을 5일에 걸쳐 실시간으로 진행되었다.



크게 보면 자료구조, 알고리즘, 컴퓨터구조, 운영체제, 데이터베이스 5과목이다.



사실 해당 내용은 2학년 ~ 3학년에 걸쳐 배웠던 내용이었는데 5일이라는 짧은 시간동안 이해하기에는 비전공자에게는 어려운 내용이 될 것으로 예상하며 들었다.



전공자인 나도 배운지 2년이 되가기에 기억이 가물가물한 부분도 있고, 아예 까먹고 있던 부분도 있었다.



특히 알고리즘부분은 주요 알고리즘 DFS, BFS의 흐름만 기억하고 구현 방법이 잘 기억나지 않았고, 아쉬웠던 부분은 탐욕 알고리즘과 같은 다른 알고리즘도 같이 다뤘으면 좋았을 것 같은데 시간 상 안됬을 것 같기에 아쉬웠다.



***


# 1. 자료구조


***



파이썬에서 다루는 자료구조부터, 어떤 알고리즘의 시간 복잡도, 공간 복잡도 등 코드의 시간적, 공간적(메모리) 효율성과 관련된 측면을 다루는 부분이다.



파이썬은 **Collections** 라이브러리의 ***deque()*** 를 통해 스택, 큐 등을 구현한다.



코딩 테스트 준비를 하다보면 스택, 큐와 관련된 문제들이 나올 때 마다 항상 사용하는 라이브러리이기 때문에 익숙하기도 했고,



개념적으로도 많이 접했기에 어려운 부분은 없었다.



시간 복잡도, 공간 복잡도도 마찬가지로 코딩 테스트를 준비하면서 문제 제한사항(조건)에 맞는 코드를 작성해야 하기 때문에 어떻게 짜야 효율적으로 코드를 작성할 수 있을까라는 고민을 항상 하므로 어렵지 않았다.



해시 테이블(딕셔너리)의 경우도 리스트로 순회하며 접근하기 보단 해시 값으로 접근하는 것이 훨씬 빠르기에 많이 사용하는 자료구조로 사용하고 있었다.



***



이 다음으로 나왔던 ***Counter, defaultdict, heapq*** 모듈들은 다소 생소했던 모듈들이었다.



**Counter**의 경우 몇 번 사용해 보긴 했지만 주로 사용하지 않는 라이브러리였고(보통 sum()집계함수를 이용), **defaultdict**의 경우 사전을 생성할 때, 생성 후 for문으로 돌며 사전에 값을 넣어줬기 때문에 굳이 초기에 기본값이 들어간 사전이 필요없다고 판단했었기에 사용하지 않았었다.



**heapq**도 거의 사용하지 않던 모듈이었는데, 작업 스케줄링과 같은 우선순위 큐 or 완전 이진 트리와 같은 많이 접할 수 있는 곳에서 사용한다는 사실을 알고 너무 고정적인(?), 내가 알고 있는 모듈, 라이브러리만 사용하려고 하는게 아닌가라는 생각이 들었다.



낯선 모듈들은 코딩 테스트 연습하면서 사용해봄으로써 손에 익을 때까지 반복해야겠다.



***


# 2. 알고리즘


***



기본적인 **sorting(정렬)**에 관련된 메서드인 sort(), sorted()를 알려주셨다.



해당 메서드들은 어느 프로젝트 or Feature Engineering 등 다양한 부분에서 굉장히 많이쓰이는 메서드들이다.



어떤 리스트를 정렬하는데 사용되는데, 매개변수들을 줌으로 다양한 방식 or 어떤 값을 기준으로 정렬할 수 있다.



다음은 **DFS, BFS**의 기초가 되는 **그래프**를 알려주셨는데 사실 그래프는 간선, 노드의 개념만 알면 쉽다.



표현도 마찬가지로 딕셔너리를 통해 연결된 노드들을 표현만 해주면 되므로 쉬운데, 이를 응용하는 DFS, BFS 알고리즘은 난이도가 꽤 있는 편이다.



나는 주로 DFS를 문제해결에 많이 사용하는 편인데, 스택 이용이 익숙해있는지라 BFS보다는 DFS에 익숙한 감이 있고, 되도록이면 DFS를 쓰는 습관이 있는데 둘 다 고루 잘 사용할 수 있어야하기에 연습이 필요할 것 같았다.



**DP(동적 계획법)**도 알게 모르게 사용하고 있었다.



사실 DP는 학부 강의 때 배웠었는데 까먹고 있었고, 이번에 강의를 듣게 되며 되새겼는데 코드를 작성하면서 꽤나 많이 사용하고 있었던 것 같다.



이전에 했던 계산을 반복하지 않음으로써 시간을 줄이는 것, 이전 졸업작품 진행할 때, 단어 사전을 생성하며 많이 썼던 방법이었다.



이러한 점이 복습의 장점이 아닐까 싶다.



내가 놓쳤던 부분이나, 미흡한 부분을 확인하고 수정해나가는 부분이 있다는 것, 한번 봐서는 내가 어느 부분에 자신이 있고, 어느 부분이 부족한지 모르기에 복습의 기회가 주어지는게 내 입장에서는 굉장히 좋은 것 같다.



***


# 3. 데이터베이스(DB)


***



학부생 시절 굉장히 재미있게 배웠던 과목이다.



**MySQL**을 통해 실습을 진행하고, 관계형 모델, 여러 키, 정규형 등 개념들까지 정보처리기사에서 자주 등장하는 문항들로 복습한지 1년도 채 되지 않았기에 기억나는 부분이 많았다.



특히 정규화 과정을 설명해 주실 때 **'도부이결다조'** 로 외웠던 기억이 났었다.



그때 당시에는 이렇게 외워서 문제를 풀려고만 했던 것 같다.



이번에 설명을 들으면서 깨닫게된 부분은 대부분 ***제 3 정규형*** 까지만 정규화를 진행한다는 점이다.



과도한 정규화는 JOIN 연산이 증가하기에 쿼리가 복잡 or 느려질 수 있기에 무작정 정규화를 진행한다고 좋은 것이 아니라 원하는 작업에 맞춰 정규화, 반정규화를 진행해야겠다.



***



그리고 존재만 알고 있었던 **sqlite3** 라이브러리로 실습하였다.



파이썬으로 DB를 사용할 수 있는 라이브러리로만 알고 있었고, 한번도 사용해보지 않았기에 유심히 보았는데, MySQL과 거의 유사한 문법과 형식을 유지하고 있었고 어렵지 않게 습득할 수 있었다.



사실 내가 주로 배웠던 DB 관련 SQL은 관계형 데이터베이스만 다뤘었다.



그런데 대규모 데이터 처리, 유연한 데이터 모델링 등의 AI에 사용되는 데이터들의 특성 상 관계형 데이터베이스만을 가지고 데이터를 관리하는데 어려움이 존재한다고 설명해 주셨고, 이에 따라 NoSQL도 잘 알아둬야겠구나 생각했다.



***



 | 특징             | RDBMS (예: MySQL, SQLite)    | NoSQL (예: MongoDB, Redis)       | 설명                                                 |
 | :--------------- | :--------------------------- | :------------------------------- | :--------------------------------------------------- |
 | **데이터 모델** | **관계형 (표)** | 문서형, Key-Value, 컬럼형, 그래프형 등 | 다양한 데이터 구조 지원 (NoSQL)                     |
 | **스키마** | **엄격함 (Fixed)** | **유연함 (Flexible) / 없음** | 데이터 구조 변경 용이 (NoSQL) 🔵                     |
 | **확장성** | 주로 수직적 확장 (Scale-up)  | 주로 **수평적 확장 (Scale-out)** | 대규모 트래픽/데이터 처리 용이 (NoSQL) 🚀              |
 | **일관성 (Consistency)** | **강력** (ACID 보장)       | **결과적 일관성 (Eventual)** 등  | 일관성 수준 조절 가능 (NoSQL, Trade-off 존재) ⚠️   |
 | **데이터 종류** | **정형 데이터** 최적화       | **비정형/반정형/정형** 모두 가능 | 다양한 형태의 데이터 저장 용이 (NoSQL) 🔵            |
 | **쿼리 언어** | **SQL** (표준)             | DB별 다양한 API / 쿼리 언어      | SQL 외 다양한 인터페이스 사용 (NoSQL)

 ***        



위의 표와 같이 요즘 핫한 LLM과 같은 언어모델을 위한 데이터들은 확실히 NoSQL로 관리하는 것이 좋은게 보인다.



이를 구현할 수 있는 툴이 ***MongoDB***이다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/MongoDB_예시.PNG?raw=true)




위와 같이 파라미터, 평가지표 등을 JSON 형식과 유사하게 저장함으로써 관계형 데이터베이스와는 다른 방식으로 저장할 수 있다.



또한 ***Vector DB***라는 비정형 데이터를 숫자 벡터로 변환하는 벡터 임베딩을 효율적으로 저장, 검색하는데 특화된 데이터베이스도 알려주셨다.



**유사도 검색**이라는 개념을 통해 LLM이 답변 생성 시, 외부 문서 벡터를 검색하여 근거있는 답변을 생성하는 상황, 즉 RAG에서 사용되는게 해당 개념인걸 알았고, 대규모 데이터에서 의미적으로 유사한 데이터를 빠르게 찾기위해 등장한 데이터베이스로 이해하였다.



해당 실습은 ***FAISS***를 통해 진행하였는데, 기존 DB와 비슷한 것 같으면서도 다르게 느껴졌다.



오히려 데이터베이스를 다룬다라는 느낌보다는 데이터를 전처리하는 과정과 유사하다고 느꼈다.



***



위의 데이터베이스 사용은 지금까지 그냥 배우니까 배우는구나 했을 뿐 실무에서 어떤식으로 이용되는지를 몰랐다.



**데이터 웨어하우스**로 다양한 데이터를 수집, 통합, 변환하고, **ETL** 프로세스로 데이터를 처리하여 품질과 신뢰성 확보, 이를 통해 빅데이터 문제를 해결하는 일련의 과정은 데이터베이스를 다루는 프로젝트 or 작업을 할 때 참고할 만한 내용인 것 같다.



***


# 4. 컴퓨터 구조 & 운영체제


***



해당 과목은 학부생 때 따로따로 배웠지만 사실상 내 생각에는 두 과목은 거의 한 과목으로 봐도 무방하다고 본다.



다루는 내용이 거의 유사하며 **컴퓨터 구조 ⊃ 운영체제**의 관계에 있다고 생각한다.



컴퓨터의 기본 구성요소인 CPU, 메모리, I/O장치 등등과 이를 연결하는 버스와 같은 기본 흐름을 배웠고, 간단하게 CPU와 GPU의 연산 차이 방식을 알려주셨다.



'그냥 AI 연산이니까 GPU를 사용해야지'가 아니라 왜 GPU를 사용하는 연산인지를 알고 사용하면 더 좋지 않을까?



나머지 컴퓨터 구조 부분은 학부생때 보다는 덜 깊게 들어갔다.



사실 해당 부분은 AI쪽에서 딥하게 다룰 이유는 많이 없었고, CPU가 어떤식으로 처리하는지, 각각의 하드웨어 장치들의 역할과 어떻게 연결되는지를 알고 메모리 절약 측면에서 코드 작성 및 데이터를 관리하면 되기에 강의를 이렇게 구성하신듯 했다.



실제로 리스트, 제네레이터의 비교와 같이 시간적, 공간적 측면의 메모리 효율과 관련된 내용이 많았다.



다음으로는 운영체제의 역할과 기능, 프로세스와 스레드의 개념을 알려주셨다.



이 부분은 처음에는 어렵게 다가왔었다. 이해할 때까지 반복해서 봤었기에 전체적인 흐름을 보고 필요한 부분을 찾는다면 사실 그렇게까지 어렵지 않은데, 전부 이해하려 했었기에 받아들이기 어려웠었다.



프로세스가 메모리에 적재되어 각각의 영역들이 처리하는 변수, CPU가 하는 역할 흐름을 설명해주셨고, 확실히 많이 보았던 내용이라 강의를 듣는데 어려움은 없었다.



다음은 리눅스 기본 명령어 및 파일 시스템 관리를 배웠는데, wsl로 리눅스 환경을 설정하여 가상환경을 관리하는 나로써는 너무나도 익숙한 명령어들이었다.



작은 프로젝트, 경진대회 등을 진행할 때 이것저것 시도해보면서 부딪혀보았기에 권한 설정, 파일 관리 등 여러 명령어들을 찾아가며 사용했었다. 이런 실습이 중요한 부분은 사실 듣는 것 보단 직접 해보며 체화하는게 훨씬 좋다는 생각이 든다.



자주 사용하던 명령어들은 익숙하지만, 리소스 모니터링에 필요한 명령어 및 라이브러리 등 내가 모르거나 낯선것들도 많기에 지속적으로 찾아봐야 할 것 같았다.



***


# 5. 네트워크


***



학습해야 할 내용이 꽤나 많아서 배울 때 애 먹었던 과목이었다.



클라이언트, 서버를 연결해주는 각각의 중계기들의 역할을 이해하고, 흐름이 어떻게 진행되는지 그리고 OSI 7계층, TCP/IP와 연결시켜 그 구조를 이해하고 각각의 계층이 어떻게 되는지 조금 까다로웠다.



이해하는데 애 먹었었던 TCP/IP를 간략하게 작성해보면...



TCP/IP : 인터넷 표준 모델로, 실제 인터넷 통신에서 사용되는 모델



| 계층 | 역할 | 관련 프로토콜 |
|------|------|----------------|
| 4. 애플리케이션 계층 | 사용자 서비스 제공 | HTTP, FTP, SMTP, DNS 등 |
| 3. 전송 계층 | 데이터 흐름 제어, 오류 제어 | TCP, UDP |
| 2. 인터넷 계층 | 라우팅, 주소 지정 | IP, ICMP, ARP |
| 1. 네트워크 인터페이스 계층 | 물리적 데이터 전송 | Ethernet, Wi-Fi 등 |



1️⃣ 네트워크 인터페이스 계층 (Network Interface Layer)

- 실제 물리적인 전송을 담당 (케이블, 와이파이 등)



- 이더넷(Ethernet), 와이파이(Wi-Fi) 같은 하드웨어 수준의 통신 방식



2️⃣ 인터넷 계층 (Internet Layer)

- IP 주소를 기반으로 패킷을 목적지로 라우팅함



- IP (Internet Protocol): 데이터 패킷의 목적지 설정



- ICMP: 에러 보고 (예: 핑 명령어)



- ARP: IP 주소 ↔ MAC 주소 변환



3️⃣ 전송 계층 (Transport Layer)

- 종단 간(end-to-end) 통신을 제공



- TCP (신뢰성 보장):



    - 연결 기반 (3-way handshake)



    - 순서 보장



    - 재전송, 오류 제어



- UDP (신뢰성 없음):



    - 연결 없이 빠름



    - 실시간 스트리밍 등에서 사용



4️⃣ 애플리케이션 계층 (Application Layer)

- 실제 사용자 서비스 제공



- HTTP: 웹 브라우징



- FTP: 파일 전송



- SMTP/POP3/IMAP: 이메일



- DNS: 도메인 이름 → IP 주소



이러한 TCP/IP모델의 각 계층의 역할을 전부 외우려고 하는 것 보다 마찬가지로 흐름을 익히는게 중요한 것 같다.



그리고 이를 파이썬 코드로 간단하게 구현하여 네이버의 IP주소를 조회하는 코드를 작성해보면 다음과 같이 작성할 수 있다.



***



```python
import socket

try:
    domain = "www.naver.com"
    ip_address = socket.gethostbyname(domain)
    print(f"'{domain}'의 IP 주소: {ip_address}")
except socket.gaierror as e:
    print(f"DNS 조회 실패: {e}")
```

<pre>
'www.naver.com'의 IP 주소: 223.130.200.219
</pre>
***



다음으로는 API를 이용하여 챗봇을 불러와 질문에 응답을 생성하는 실습을 진행하였다.



모델은 gemini를 이용하였는데, 이를 이용하여 작은 프로젝트를 진행할 때 유용하게 사용할 수 있을 것 같았다.



***



```python
import google.generativeai as genai

genai.configure(api_key="AIzaSyB3NbQa8JiHYqTCNdIAWs_bye1FRnFumK0")

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

response = model.generate_content("챗지피티 vs 제미나이, 간단하게 요약")

print(response.text)
```

<pre>
ChatGPT와 Gemini는 모두 거대 언어 모델(LLM)이지만, 몇 가지 주요 차이점이 있습니다.

* **ChatGPT:** 오픈AI 개발, 주로 대화 및 텍스트 생성에 특화.  다양한 스타일의 텍스트 생성에 능숙하지만, 최신 정보 접근이 제한적일 수 있습니다.

* **Gemini:** 구글 개발, ChatGPT보다 다양한 모달리티(텍스트, 코드, 이미지 등)를 처리할 수 있는 멀티모달 모델.  최신 정보 접근성이 더 뛰어날 가능성이 높습니다.


간단히 말해, ChatGPT는 대화형 텍스트 생성에 강점을 보이고, Gemini는 다양한 형태의 정보를 처리하는 능력이 뛰어납니다.  어떤 모델이 더 "좋은"지는 사용 목적에 따라 다릅니다.

</pre>
***



또한 ***WindSurf*** IDE를 알려주셨는데 이게 굉장히 좋았다.



VSCode와 연동시켜 지금까지 했던 프로젝트, 코드 등을 그대로 불러와 사용할 수 있었고, 해당 코드를 AI를 이용하여 문제점 분석, 효율성 분석 등을 통해 코드를 리팩토링하는데 굉장히 유용하였고, 또한 어떤 코드의 뼈대를 생성할 때도 유용하게 사용할 수 있었다.







![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/Windsurf_활용.PNG?raw=true)







***

