---
layout: single
title:  "Pwntools 설치 및 GCC 설치"
categories: 정보보안
tag: [python, pwn, hacking]
toc: true
author_profile: true
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


pwntools는 파이썬으로 익스플로잇을 수행할 수 있도록 도와주는 라이브러리이다.



일반적으로, 어떤 대상의 컴퓨터를 어택할 때에는 어셈블리어를 이용하여 익스플로잇을 수행한다.



하지만 어셈블리어는 기계와 친화적인 언어로, 인간이 이해하기에는 다소 난해한 부분이 존재한다.



따라서 고급 언어로 인간의 이해를 돕기 위해, 여러 라이브러리(도구)들이 존재하는데, 그 중 하나가 pwntools이다.


## 칼리 리눅스 내 pwntools 설치


칼리 리눅스 내 pwntools를 설치하는 방법은 간단하다.



칼리 리눅스 자체가 해킹에 특화된 리눅스 커널이기 때문에 터미널에 몇 개의 명령어만 실행해 주면 된다.


아래의 명령어들을 터미널을 열어 설치해주면 끝난다.


```bash

$ apt-get update

$ apt-get install python3 python3-pip python3-dev git libssl-dev libffi-dev build-essential

$ python3 -m pip install --upgrade pip

$ python3 -m pip install --upgrade pwntools

```


## 윈도우 내 pwntools 설치


윈도우에서 pwntools를 다양한 개발환경에서 실행하기 위해 설치를 먼저 해야한다.



cmd or powershell을 열어 다음과 같이 하나 작성 후 설치한다.


```bash

pip install pwntools

```


이후 개발환경에 들어가서 pwntools이 정상적으로 작동하는지 확인한다.



```python
from pwn import *
```

또한 binutils를 설치해야하는데, 이는 윈도우용 바이너리를 다운로드한다.



https://sourceforge.net/projects/mingw-w64/files/



해당 사이트 접속 후 자신의 컴퓨터에 맞는 최신 버전 바이너리를 설치한다.


다운로드한 파일을 압축 해제 후 해당 파일을 C:\Program Files(x86)에 복사한다.


시스템 환경 변수 편집 → 고급 → 환경 변수에 들어간다.



시스템 변수에서 Path에 해당하는 값을 편집으로 다음과 같이 하나 추가한다.


![image.jpg](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/환경_변수_편집.png?raw=true)


터미널을 열어 다음과 같은 명령어를 실행한다.



두 가지중 하나의 명령을 실행하면 된다.


```bash

gcc -v

```


```bash

g++ -v

```


다음과 같은 창이 뜬다면 설치 완료

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/gcc_.버전_확인.png?raw=true)

pwntools과 바이너리 설치가 모두 끝났다.



이제 여러 개발환경에서 pwntools 및 GCC로 C언어 컴파일이 가능하다.

## GNU 설치

위의 과정을 거친 후에도 binutils에 의한 에러가 뜬다면 직접 GNU를 설치해준다.

 https://www.gnu.org/software/binutils/

 해당 url을 접속하여 binutils를 자신의 컴퓨터에 맞는 최신 버전을 다운 받아 위의 환경 변수 설정과 똑같이 적용한다.


## pwntools 기본 함수


### 익스플로잇 대상 선택


```python

p.process(address)

p.remote(address)

```


process()함수는 로컬 바이너리를 대상으로 익스플로잇을 수행한다.



process()는 실제 서버를 대상으로 익스플로잇 하기 전, 미리 테스트 해보는 것으로 로컬 바이너리 폴더 등을 대상으로 지정할 수 있다.



p.remote()함수는 원격 서버를 대상으로 익스플로잇을 수행한다.



remote()는 실제 사용하는 서버에서 익스플로잇을 하기 위해 사용한다.



따라서 어떤 서버를 익스플로잇하고자 할 때에는 remote()함수로 직접적인 익스플로잇이 가능하다.



### send() 함수


```python

send('A')

sendline('A')

sendafter('Lee''A')

sendlineafter('Lee','A')
```


기본적으로 send가 들어가는 함수는 데이터를 프로세스에 전송하기 위해 사용된다.



send('A') : 어떤 파일에 'A'를 입력한다.



sendline('A') : 파일에 'A' + '\n'을 입력한다.



sendafter('Lee','A') : 파일이 'Lee' 출력 시, 'A'을 입력한다.



sendlineafter('Lee','A') : 파일이 'Lee' 출력 시, 'A' + '\n'을 입력한다.


### recv() 함수


```python

recv([Byte])

recvline()

recvn([Byte])

recvuntill('Lee')

recvall()

```



recv()함수는 프로세스에서 데이터를 받기 위한 함수이다.



recv([Byte]) : 파일이 출력하는 데이터를 최대 [Byte]까지 받는다.



recvline() : 파일이 출력하는 데이터를 개행문자를 만날 때 까지 받는다.



recvn([Byte]) : 파일이 출력하는 데이터를 [Byte]만 받는다.



recvuntil('Lee') : 파일이 출력하는 데이터를 'Lee'가 출력될 때 까지 받는다.



recvall() : 파일이 출력하는 데이터를 프로세스가 종료될 때까지 받는다.


### 패킹, 언패킹


```python

p32()

p64()

```


어떤 값을 리틀 엔디언의 바이트 배열로 변경한다.



리틀 엔디언이란, CPU에서 메모리에 데이터를 저장하는 방식이다.



가장 낮은 주소에 최하위 비트를 저장, 그 다음 주소에 두 번째로 낮은 비트 저장... 이런 식으로 진행된다.



예를 들어, '0x12345678'을 리틀 엔디언으로 나타내면 다음과 같다.



Address     |   Byte Value



0x000000    |   0x78



0x000004    |   0x56



0x000008    |   0x34



0x000010    |   0x12





리틀 엔디언과 반대되는 표현 방식으로는 빅 엔디언이다.



빅 엔디언은 가장 낮은 주소에 최상위 비트를 저장한다.



Address     |   Byte Value



0x000000    |   0x12



0x000004    |   0x34



0x000008    |   0x56



0x000010    |   0x78



여기서 p뒤에 붙는 32, 64는 사용하는 컴퓨터의 비트이다.



```python

u32()

u64()

```


p32() 및 p64() 함수의 역의 과정이다.



따라서 p32(), p64()는 패킹, u32(), u64()는 언패킹 과정이다.


### 쉘 획득


```python

interactive()

```


익스플로잇 과정에서 쉘 획득 or 출력 확인에 사용되는 함수이다.



익스플로잇이 성공적으로 된다면, 중요한 정보등을 얻을 수 있다.


### 이외 함수들 참조


https://docs.pwntools.com/en/stable/shellcraft/amd64.html



위의 주소는 pwntools에 관련된 함수들을 설명한다.

