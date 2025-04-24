---
layout: single
title:  "[프로그래머스] k진수에서 소수 개수 구하기"
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


그냥 문제 설명을 보고 그대로 코드를 구현했다.



먼저 작업해 준 것은 k진수 변환 함수와 소수를 검사하는 함수 두 개를 구현했다.



```python
# 10진수 → k진수 변환 함수

def convert_k(n, k) :
    k_number = ''
    
    while n > 0 :
        quotient = n // k
        reminder = n % k
        n = quotient
        k_number += str(reminder)
    
    return k_number[::-1]
```


```python
# 소수 검사 함수

import math

def check_prime(n) :
    if n == 1 :
        return False
    
    for i in range(2, int(math.sqrt(n)) + 1) :
        if n % i == 0 :
            return False
    return True
```

10진수를 k진수로 변환하는 과정은 주어진 값을 k로 나누어 나머지를 계속 더하면 된다.



소수 검사 함수는 제곱근을 활용하여 시간복잡도를 줄였다. (n이 최대 1,000,000이라 k진수 변환 후 전부 검사하면 시간 초과날 것 같았다.)



***


그리고서 문제에서 주어진 대로...



1. 0이 아니면 **number_string**에 추가



2. 0이 나오면 지금까지 **number_string**에 있던 값이 소수인지 검사



3. 1, 2 과정 반복



4. 마지막이 만약 0 이 아니라면(EX: 912093) **number_string**에 있던 값 소수 검사


# 2. 코드 구현



```python
import math

def convert_k(n, k) :
    k_number = ''
    
    while n > 0 :
        quotient = n // k
        reminder = n % k
        n = quotient
        k_number += str(reminder)
    
    return k_number[::-1]

def check_prime(n) :
    if n == 1 :
        return False
    
    for i in range(2, int(math.sqrt(n)) + 1) :
        if n % i == 0 :
            return False
    return True
    

def solution(n, k):
    answer = 0
    number_string = ''
    
    converted_number = convert_k(n, k)
    
    for i in range(len(converted_number)) :
        if converted_number[i] != '0' :
            number_string += converted_number[i]
        else :
            if number_string != "" :   
                if check_prime(int(number_string)) :
                    answer += 1
                    number_string = ''
                    continue
                else :
                    number_string = ''
                    continue
    
    if '0' not in number_string :            
        if number_string != "" and check_prime(int(number_string)) :
            answer += 1
    
    return answer
```


```python
solution(50, 10)
```

<pre>
1
</pre>

```python
convert_k(1_000_000, 3)
```

<pre>
'1212210202001'
</pre>