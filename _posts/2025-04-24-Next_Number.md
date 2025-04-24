---
layout: single
title:  "[프로그래머스] 다음 큰 숫자"
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


문제가 쉬웠다.



1. 주어진 n값을을 2진수로 변환하고, 1의 갯수를 센다.



2. n += 1 을 해주며 1의 과정과 똑같은 과정을 반복, n과 1의 갯수가 같은지 비교



3. 같다면 반복문을 멈추고 return


# 2. 코드 구현



```python
def solution(n):
    binary_number = bin(n)[2:]
    one_count_n = binary_number.count("1")
    
    while True :
        n += 1
        if one_count_n == bin(n)[2:].count("1") :
            break
    return n
```


```python
solution(78)
```

<pre>
83
</pre>