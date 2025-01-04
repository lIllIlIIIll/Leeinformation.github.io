---
layout: single
title:  "Stack Link & Dynamic Link"
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


## 링크


많은 프로그래밍 언어에서 컴파일의 마지막 단계로, 프로그램에서 어떤 라이브러리의 함수를 사용한다면, 호출된 함수와 실제 라이브러리의 함수가 링크 과정에서 연결된다.


리눅스에서 C언어로 작성된 코드는 전처리, 컴파일, 어셈블의 과정을 거쳐 

ELF 형식을 갖춘 오브젝트 파일로 번역된다.


```bash

$ gcc -c hello-world.c -o hello-world.o

```


위의 명령어로 hello-world.c를 어셈블할 수 있다.


C 코드는 다음과 깉다.



```C

#include <stdio.h>



int main() {

    puts("Hello, world!");

    return 0;

}

```


오브젝트 파일은 실행 가능한 형식을 갖추고 있다.



하지만 라이브러리 함수들의 정의가 어디 있는지 알 수 없기 때문에 실행은 불가능하다.


```bash

$ readelf -s hello-world.o | grep puts

```


위의 명령어를 실행해 보면 puts의 선언이 <stdio.h>에 있어 심볼로는 기록이 되어있지만, 이에 대한 내용은 하나도 기록되어 있지 않다.

![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/링크_readelf.png?raw=true)


이러한 심볼과 관련된 정보들을 찾아 최종 실행 파일에 기록하는 것이 링크가 하는 역할이다.


이제 hello-world.c를 컴파일 하고, 다음 명령어를 통해 링크되기 전과 비교한다.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/링크_컴파일.png?raw=true)


libc에서 puts의 정의를 찾아 연결된다.



여기서 libc를 같이 컴파일하지 않았는데, libc에서 해당 심볼을 탐색한 것은 libc가 표준 라이브러리 경로에 포함되어 있기 때문이다.



표준 라이브러리 경로는 다음과 같은 명령어로 확인할 수 있다.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/표준_라이브러리_경로.png?raw=true)


결국 프로그램에서 puts를 호출하는 과정은 다음과 같다.



1. puts 호출



2. libc에서 puts 코드 탐색



3. puts 코드 실행


## 라이브러리와 링크


라이브러리는 크게 동적 라이브러리와 정적 라이브러리로 구분된다.



동적 라이브러리를 링크하는 것을 동적 링크, 정적 라이브러리를 링크하는 것을 정적 링크라한다.


### 동적 링크


동적 링크된 바이너리를 실행하면 동적 라이브러리가 프로세스의 메모리에 매핑된다.



그리고 프로그램 실행 중에 라이브러리의 함수를 호출하면 매핑된 라이브러리에서 호출할 함수의 주소 탐색, 이 후 함수를 실행한다.



간단히 비유하면, 도서관에서 원하는 책의 위치를 찾고, 그 책에서 정보를 얻는 과정과 유사하다고 볼 수 있다.


### 정적 링크


정적 링크된 바이너리를 실행했을 때에는 바이너리에 정적 라이브러리의 필요한 모든 함수가 포함된다.



따라서 해당 함수를 호출할 때, 라이브러리를 참조하는 것이 아닌 자신의 함수를 호출하는 것처럼 호출할 수 있다.



해당 방법은 여러 바이너리에서 라이브러리를 사용하면 해당 라이브러리의 복제가 여러번 이루어지기 때문에 용량 낭비가 생긴다.


### 동적 링크 VS 정적 링크


앞의 hello-world.c를 컴파일한다.



정적 컴파일은 static으로, 동적 컴파일은 dynamic으로 생성한다.


```bash

$ gcc -o static hello-world.c -static

$ gcc -o dynamic hello-world.c -no-pie

```


이제 동적 링크와 정적 링크를 비교해본다.


#### 용량


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/용량_비교.png?raw=true)


정적으로 생성된 파일의 용량이 동적으로 생성된 파일의 용량보다 50배 더 많은 용량을 차지하는 것을 볼 수 있다.


#### 호출 방법


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/호출_방법.png?raw=true)


위의 사진에서 왼쪽이 static, 오른쪽이 dynamic이다.



살펴보면, static에서는 puts가 있는 주소를 직접 호출하는 반면,



dynamic에서는 puts의 plt 주소를 호출하는 것을 볼 수 있다.



동적 링크된 바이너리는 함수의 주소를 라이브러리에서 찾아야 하기 때문에 이러한 차이가 발생하고, plt는 이 과정에 사용되는 테이블이다.


## PLT & GOT


두 테이블은 라이브러리에서 동적 링크된 심볼의 주소를 찾을 때 사용하는 테이블이다.



바이너리가 실행되면, 라이브러리가 임의의 주소에 매핑되고, 이 상태에서 라이브러리 함수를 호출하면, 함수의 이름을 바탕으로 라이브러리에서 심볼 탐색, 해당 함수의 정의를 발견하면 그 주소로 실행 흐름을 옮긴다.



만약 반복적으로 호출되는 함수가 있을 때, 이러한 과정을 매번 반복한다면 이는 비효율적이다.



그렇기에 ELF는 GOT라는 테이블에 함수의 주소를 테이블에 저장한다.



저장된 함수의 주소는 필요할 때 꺼내서 사용하게 된다.


```c

#include <stdio.h>



int main() {

    puts("Resolving address of 'puts'.");

    puts("Get address from GOT");

}

```


위의 C 코드를 컴파일하여 어떠한 방식으로 작동되는지 살펴 본다.


```bash

$ gdb ./got

pwndbg> entry

pwndbg> got

```


위의 명령어들을 입력하면 GOT의 상태를 보여준다.


```bash

pwndbg> plt

```


puts의 GOT 엔트리에는 아직 puts의 주소를 찾기 전이기 때문에, 함수 주소 대신 .plt 섹션 어딘가의 주소를 알려준다.


밑의 사진을 보자.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/got&plt.png?raw=true)


got 명령어를 실행하면 puts의 GOT 엔트리인 0x404000에는 함수의 주소 대신 plt 섹션 어딘가의 주소 0x401036이 저장되어있다.



이 주소에 push로 puts@plt가 저장된다.


### 시스템 해킹 관점에서의 PLT & GOT


PLT 및 GOT는 동적 링크된 바이너리에서 라이브러리 함수의 주소를 찾고 기록할 때 사용된다.



해커의 관점으로 보았을 때, PLT에서 GOT를 참조하여 실행 흐름을 옮길 때, GOT의 값을 "검증하지 않는다"는 보안 취약점이 존재한다.



만약 puts의 GOT엔트리에 저장된 값을 공격자가 임의로 변경이 가능하다면, puts가 호출될 때 공격자가 원하는 코드를 실행시킬 수 있게된다.



이러한 방법으로 GOT 엔트리에 임의의 값을 OVerwrite하여 실행 흐름을 변조하는 공격 기법을 GOT Overwrite라고 한다.

