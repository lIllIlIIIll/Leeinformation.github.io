---
layout: single
title:  "Stack Buffer Overflow & Stack Canary"
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


## 스택 버퍼 오버플로우


스택 버퍼 오버플로우란 스택의 버퍼에서 발생하는 오버플로우이다.



스택 버퍼 오버플로우를 이해하기 전 버퍼와 오버플로우가 무엇인지 먼저 알아야한다.


### 버퍼


데이터가 목적지로 이동되기 전 보관되는 임시 저장소이다.



만약, 데이터 처리속도가 다른 두 장치가 존재할 때, 해당 두 장치 사이에 오가는 데이터를 임시로 저장해 주는 역할을 한다.



데이터 처리 속도가 1인 장치 1과 데이터 처리속도가 2인 장치 2가 있다고 가정해보자.



장치 1과 장치 2가 각각 다른 데이터를 동시에 처리할 때, 장치 2의 데이터 처리속도가 빠르기 때문에, 장치 1에서 프로그램에서 수용되지 못한 데이터는 모두 유실된다.



키보드에서 12345678을 입력했는데 1234만 프로그램에 전달될 수 있다는 뜻이다.



따라서 버퍼라는 임시 저장소를 사용하여, 송신 측에서 데이터를 버퍼로 보내고, 수신 측에서는 버퍼에서 데이터를 꺼내 사용한다.



따라서, 버퍼가 가득 찰 때 까지는 데이터 유실 없이 데이터 통신이 가능하다.



- 스택 버퍼 : 스택에 있는 지역 변수



- 힙 버퍼 : 힙에 할당된 메모리 영역


### 버퍼 오버플로우


위에서 설명하였던 버퍼가 넘치는 것을 의미한다.



예를 들어, char 배열이 10 byte의 크기를 가질 때, 20 byte 크기의 데이터가 들어가면 이 때 오버플로우가 발생한다.



일반적으로, 버퍼는 메모리상에서 연속해서 할당되기 때문에, 버퍼 오버플로우 발생 시 뒤의 버퍼 값이 조작될 위험이 존재한다.



밑의 그림을 통해 더 쉽게 이해가 가능하다.

![image.jpeg](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/스택_구조.png?raw=true)


따라서, 버퍼 오버플로우는 일반적으로 어느 메모리 영역에서 발생하여도, 보안에 있어 큰 위협으로 이어진다.


### 데이터 변조


```c

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

int check_auth(char *password) {

    int auth = 0;

    char temp[16];

    

    strncpy(temp, password, strlen(password));

    

    if(!strcmp(temp, "SECRET_PASSWORD"))

        auth = 1;

    

    return auth;

}

int main(int argc, char *argv[]) {

    if (argc != 2) {

        printf("Usage: ./sbof_auth ADMIN_PASSWORD\n");

        exit(-1);

    }

    

    if (check_auth(argv[1]))

        printf("Hello Admin!\n");

    else

        printf("Access Denied!\n");

}

```


코드를 보며 어떤 방식으로 데이터 변조가 진행되는지 알아보겠다.



main 함수에서 check_auth 함수에 argv[1]인자를 전달, 반환값을 받아와서 1이라면 "Hello Admin!"을 출력하고, 이외의 경우 "Access Denied!"를 출력한다.



check_auth 함수를 살펴보면, 16 byte 크기의 temp 버퍼에 입력받은 password를 복사한 뒤, SECRET_PASSWORD와 비교하여 같다면 auth를 1로 설정하고 반환한다.



여기서 strncpy 함수를 통해 버퍼를 복사할 때, password의 크기만큼 복사하는데 만약, argvv[1]에 16 byte가 넘는 문자열이 전달된다면 문자열 전부 복사되어 스택 버퍼 오버플로우가 발생한다.



일반적으로 C 언어에서는 인자 전달 및 함수 호출에 사용되는 스택 메모리가 역순으로 배치되는데, temp 버퍼 뒤에 auth가 존재하므로 temp에 오버플로우를 발생시켜 auth 값을 임의의 값으로 변조할 수 있다.



따라서 check_auth 함수의 인증을 무시하고 항상 참으로 만든다.


### 데이터 유출


C 언어에서 정상적인 문자열은 Null 바이트로 종결되고, Null 바이트를 문자열의 끝으로 인식한다.



만약 버퍼에 오버플로우를 발생시켜 버퍼와 버퍼 사이의 Null 바이트를 제거한다면, 다른 버퍼의 데이터를 읽을 수 있다.



이를 통해 중요한 데이터를 유출시키거나 보호기법을 우회할 수 있다.


### 실행 흐름 조작


프로그램의 실행 흐름에서 함수 호출 시, 반환 주소를 스택에 쌓고, 함수에서 반환될 때, 이를 꺼내 원래의 실행 흐름으로 돌아간다.



여기서 스택 버퍼 오버플로우로 반환 주소를 조작하면 프로세스의 실행 흐름을 조작할 수 있다.


## 스택 카나리


스택 카나리는 스택 버퍼 오버플로우로부터 반환 주소를 보호하는 기법이다.



함수의 프롤로그에서 스택 버퍼와 반환 주소 사이에 임의의 값을 넣어 함수의 에필로그에서 해당 값의 변조를 확인한다.



여기서 스택 버퍼와 반환 주소 사이 임의의 값을 카나리 값이라 하고, 이 카나리 값의 변조가 확인되면 프로세스는 강제 종료된다.



따라서 스택 버퍼 오버플로우로 반환 주소를 덮을 때 카나리 값을 먼저 덮어야 한다.


### 카나리 활성화 & 비활성화


기본적으로 우분투 환경에서의 gcc는 스택 카나리가 적용되어 바이너리를 컴파일한다.



카나리 없이 컴파일 하고 싶다면 다음과 같이 컴파일하면 된다.


```bash

$ gcc -o no_canary canary.c -fno-stack-protector

```


위와 같이 -fno-stack- 옵션을 추가하여 카나리 없이 컴파일이 가능하다.



카나리 없이 컴파일 후 스택 버퍼 오버플로우를 발생시키면, **Segmentation fault**가 발생한다.

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/no_canary.png?raw=true)

이제 카나리를 적용하여 컴파일을 한다.

``` bash

$ gcc -o canary canary.c

```

해당 방법으로 컴파일 하였을 때, 카나리가 적용이 안된다면 다음과 같은 방법으로 카나리를 적용시킬 수 있다.



```bash

$ gcc -o canary canary.c -fstack-protector

```

위의 방법으로 컴파일 시 카나리가 적용되어 컴파일 된다.


다시 카나리를 적용하여 컴파일을 하고 스택 버퍼 오버플로우를 발생시키면


**stack smashing detected** 와 **Aborted** 에러가 발생한다.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/canary.png?raw=true)


위의 사진과 같이 에러가 발생하는데, **Aborted**에러는 안 나타날 수 있다.


### 카나리 분석


```bash

$ gdb -q ./canary

pwndbg> break *main+8

pwndbg> r

```


프롤로그 코드에 중단점을 지정하고 바이너리를 실행시킨다.

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/canary_fs.png?raw=true)


위의 사진에서 main+8은 fs:[0x28]의 값을 읽어 rax 레지스터에 저장한다.



fs는 세그먼트 레지스터의 일종으로, 프로세스가 시작할 때 랜덤 값을 저장한다.



따라서 rax에는 랜덤으로 생성된 값이 저장된다.



이제 코드 한 줄을 실행해 본다.


rax를 보면 랜덤으로 생성된 값으로 변경된 것을 볼 수 있다.

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/canary_rax.png?raw=true)


이제 중단점을 main+50에 설정하고 바이너리를 계속 실행시킨다.



해당 부분을 보면, [rbp-8]에 저장한 카나리를 rdx로 옮기고, fs:[0x28]에 저장된 카나리를 빼서 동일하면 결과값이 0이되면서 main 함수는 정상적으로 반환된다.



하지만, 두 값이 동일하지 않다면, __stack_chk_fail이 호출되며 프로그램이 강제 종료된다.

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/canary_je.png?raw=true)


### 카나리 생성 과정


위에서 적었듯이, 카나리는 fs를 통해 생성된다.



여기서 fs는 TLS를 가리키는데, fs의 값을 알면 TLS의 주소를 알 수 있다.



하지만 fs의 값은 특정 시스템 콜을 사용해야 조회 or 설정이 가능하다.



따라서 fs의 값을 설정할 때 호출되는 arch_prct; 시스템콜에 중단점을 설정하여 어떻게 설정되는지 알아본다.


```bash

$ gdb -q ./canary

pwndbg> catch syscall arch_prctl

pwndbg> r

```


catchpoint에 도달하였을 때, rdi의 값은 0x1002로 해당 값은 ARCH_SET_TS의 상숫값이다.



시스템콜이 요청 인자 순서에 따라 다음 rsi 값을 보면 0x7ffff7faa740이고, 이 프로세스는 TLS를 0x7ffff7faa740에 저장할 것이고, fs는 이를 가리킨다.



카나리가 저장될 fs+0x28을 보면 아직 어떠한 값도 설정되어있지 않다.

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/canary_TLS.png?raw=true)


```bash

pwndbg> watch *(0x7ffff7faa740+0x28)

```

위의 코드를 실행하여 watchpoint를 설정하여 프로세스를 진행시킨다.


이제 TLS+0x28의 값을 조회하면 카나리 값이 설정된 것을 볼 수 있다.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/canary_value.png?raw=true)


### 카나리 우회


1. 무차별 대입



x64 아키텍쳐에서는 8바이트의 카나리가, x86 아키텍쳐에서는 4바이트의 카나리가 생성된다.



각각의 카나리에는 Null 바이트가 포함되어있으므로 실제 7바이트, 3바이트의 랜덤한 값으로 구성된 카나리가 생성된다.



무차별 대입으로 각각의 아키텍쳐에서 카나리 값을 알아내기 위해 x64에서는 최대 256^7번을, x86에서는 256^3번의 연산이 필요하다.



즉, 무차별 대입으로 카나리값을 알아내는 것은 현실적으로 불가능에 가깝다.


2. TLS 접근



카나리는 TLS에 전역변수로 저장되고, 매 함수마다 이를 참조해서 사용한다.



만약 실행 중 TLS 주소를 알 수 있고, 임의의 주소에 대한 읽기 or 쓰기가 가능하다면 TLS에 설정된 카나리 값을 읽거나 조작할 수 있다.



이 후, 스택 버퍼 오버플로우를 수행할 때 알아낸 카나리 값 or 조작한 카나리 값으로 스택 카나리를 덮어 카나리 검사를 우회할 수 있다.

