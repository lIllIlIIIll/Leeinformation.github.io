---
layout: single
title:  "[프로그래머스] 유사 칸토어 비트열"
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


문제의 조건을 확인해보면 다음과 같다.



$$1 ≤ n ≤ 20$$



$$1 ≤ l, r ≤ 5^n$$



$$l ≤ r < l + 10,000,000$$



이는 유사 칸토어 비트열로 변환하고 구간 내 1의 갯수를 센다는 것은 불가능함을 보여준다.



n이 20이면 $5^{20}$, 이는 필연적으로 규칙을 찾아야함을 의미한다.



그렇기에 먼저 1을 11011로 치환하는 과정을 살펴보았다.



***



가운데, 즉 index가 2일 때 0이 되고 나머지는 1로 치환되는 것을 볼 수 있다.



즉 첫 번째 5개의 수와 두 번째 5개의 수에 대해서만 어떤 수를 5로 나눴을 때 나머지가 2라면 0으로 치환된다.



나머지 11번째 수 부터 어떻게 해야할 지 고민하던 중 다음과 같은 규칙을 발견했다.



| 데이터                        | 표현식     | n 값  |

|-----------------------------|-----------|-------|

| 1     1     0     1     1       | $4^1$     | n = 1 |

| 11011 11011 00000 11011 11011   | $4^2$     | n = 2 |

| 5<sup>3</sup>개                | $4^3$     | n = 3 |



n = 2 일때를 보면 각각의 인덱스가 가지는 수를 보면...



- 2, 7, 10, 11, 12, 13, 14, 17, ... : 1



- 그 외 나머지 : 0



해당 수의 규칙이 처음에는 보이지 않았다. 하지만 문제의 제한사항에서 $5^n$을 보고 5진법으로 변환하면 어떨까라는 생각을 하게 되었고



실제로 5진법 변환함수를 만들어 해당 수들을 넣어보았다.



```python
def convert_5(n) :
    five_number = ''
    
    while n > 0 :
        quotient = n // 5
        reminder = n % 5
        n = quotient
        five_number += str(reminder)
        
    return five_number[::-1]

convert_5(7)
```

<pre>
'12'
</pre>

```python
list = [2, 7, 10, 11, 12, 13, 14, 17]

for value in list :
    print(convert_5(value))
```

<pre>
2
12
20
21
22
23
24
32
</pre>
이제 보일텐데, 해당 수들은 5진법으로 변환했을 때, 전부 2라는 수를 가지고 있다.



따라서 이에 맞춰 주어진 구간의 값들을 전부 5진법으로 변환하고 안에 2가 존재한다면 count를 1 늘려주면 되겠구나 생각했다.



하지만 이런 방식도 시간 초과가 떴다.



그래서 생각했던게 굳이 5진법으로 전부 표현하지 않고 변환 중 2가 나오면 break를 걸어주면 되겠다하고 실제로 코드에 옮겼다.


# 2. 코드 구현



```python
def convert_5(n) :
    five_number = ''
    
    while n > 0 :
        quotient = n // 5
        reminder = n % 5
        n = quotient

        if reminder == 2 :
            break
        
    else :
        return True
        
    return False

def solution(n, l, r):
    count = 0
    for i in range(l-1, r) :
        x = convert_5(i)
        if x :
            count += 1
            
    return count
```


```python
solution(2, 4, 17)
```

<pre>
8
</pre>