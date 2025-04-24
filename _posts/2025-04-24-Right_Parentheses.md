---
layout: single
title:  "[프로그래머스] 올바른 괄호"
categories: Coding_Test
tag: [python, coding]
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


# 1. 생각의 흐름


자료구조를 공부할 때 맨 처음 접할 수 있는 문제 유형이다.



구현 방식은 **stack**으로 구현하였으며, 문제에서는 '***()***'만 나온다고 하였지만, 일반적인 상황에서는 수식 or 텍스트와 같이 사용되므로, 모든 상황에서 괄호가 정상적으로 사용되었는지 체크한다.



- 구현 순서



    1. "("을 stack에 넣고 나머지 문자 or 숫자 등은 stack에 넣지 않는다.



    2. ")" 등장 시 stack의 마지막(최근)에 넣은 값이 "("일 경우 stack에서 해당 값을 뺀다.



    3. 주어진 문자열에 대해 모든 과정이 끝났을 때, stack이 비어있다면 True를 반환한다.


# 2. 코드 구현



```python
from collections import *

def solution(s):
    stack = deque()
    for char in s :
        if char == "(" :
            stack.append(char)
        elif char == ")" :
            if stack and stack[-1] == "(" :
                stack.pop()
                continue
            stack.append(char)
        else :
            pass
            
    if stack :
        return False
    else :
        return True
```


```python
solution("(()()())")
```

<pre>
True
</pre>
# 3. 다른 방식의 풀이


코딩 테스트 스터디원이 구현한 다른 방식이다.



"(" 등장 시 count를 1 더하고, ")" 등장 시 count를 -1 한다.



만약 for문 도중 count가 음수가 된다면 이는 괄호가 짝이 맞지 않음으로 False를 반환한다.



또한, 모든 과정이 끝났을 때, count가 0이라면 짝이 맞는 것 이므로 True를 반환, 이외에는 False를 반환한다.



```python
def solution(s):
    count = 0
    for char in s:
        if char == '(':
            count += 1
        else:  # char == ')'
            count -= 1
        if count < 0:
            return False
    return count == 0
```
