---
layout: single
title:  "[프로그래머스] 비밀 코드 해독"
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


문제 조건에서 n값이 생각보다 작았다. (30 이하)



이는 완전 탐색을 해도 시간이 모자라지 않을 수도 있다는 점을 시사한다.



그래서 조합을 사용했다.



모든 경우의 수를 전부 나열한 다음 조건에 맞는 조합을 골라 갯수를 센다.



1. 1부터 n까지의 수로 5개로 만들 수 있는 모든 조합을 구한다.



2. 그 조합과 q를 비교하여 ans개 만큼이 값이 같다면 answer_list에 넣는다.



3. 모든 탐색을 끝난 뒤 answer_list의 길이를 구한다.


# 2. 코드 구현



```python
from itertools import combinations

def solution(n, q, ans) :
    n_list = list(range(1, n+1))
    answer_list = []
    
    for combination in combinations(n_list, 5) :
        for input, a in zip(q, ans) :
            if len(set(combination).intersection(input)) != a :
                break
        else :
            answer_list.append(combination)
    
    return len(answer_list)
```


```python
solution(10,
         [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [3, 7, 8, 9, 10], [2, 5, 7, 9, 10], [3, 4, 5, 6, 7]],
         [2, 3, 4, 3, 3])
```

<pre>
[(3, 4, 7, 9, 10), (3, 5, 7, 8, 9), (3, 5, 7, 8, 10)]
</pre>
<pre>
3
</pre>