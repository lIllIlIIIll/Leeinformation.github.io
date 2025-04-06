---
layout: single
title:  "[프로그래머스] 봉인된 주문"
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


# 1. 10진법 → 26진법 / 26진법 → 10진법 변환


문제를 접했을 때 영어 알파벳은 26개이므로 26진법을 활용해야겠구나 생각했다.



일단 먼저 주어진 문자를 수로 표현하는 코드를 구현하였다.



26진법을 이용하여 주어진 문자가 몇 번째 숫자인지를 계산한다.



```python
def convert_26(spell) :
    result = 0
    location = 1
    
    for i in range(len(spell)-1, -1, -1) :
        result += (ord(spell[i]) - 96) * location
        location *= 26
        
    return result
```


```python
convert_26('python')
```

<pre>
201883748
</pre>
다음으로는 숫자로 표현되어 있는 값을 문자로 변환해주어야 하는데 여기서 애 좀 먹었다.



처음에는 **num**이 26일 때, 26보다 작을 때, 26보다 클 때를 구분하여 케이스를 나눴지만 너무 복잡해졌고



다시 생각해보니 나눌 필요가 없었다. (마지막에 몫을 더하는 방식을 이용했었다...)



**while문**으로 26을 계속해서 나눠 몫과 나머지를 계산하고, 이를 문자열로 변환하여 더해주면 되었다.



또한 마지막으로 26으로 나눈 나머지를 한번 더 더해주면 굳이 몫을 더할 필요 없이 숫자가 문자로 변환되었다.



```python
def convert_10(num) :
    result = ''
        
    while num // 26 > 0 :
        quotient = num // 26
        reminder = num % 26
        num = quotient
        
        if reminder == 0 :
            result += 'z'
            num -= 1
        
        else :
            result += chr(reminder+96)
            
    reminder = num % 26
    
    if reminder == 0 :
        return result[::-1]
                    
    result += chr(reminder+96)
        
    return result[::-1]
```


```python
convert_10(201883748)
```

<pre>
'python'
</pre>
# 2. 최종 코드 구현


문자를 숫자로, 숫자를 문자로 변환하는 함수가 완성됬으므로 나머지는 간단하다.



> 과정



    1. **spell_converted_list** 라는 리스트에 금지된 주문의 문자에 해당하는 수를 넣고 정렬한다.

    2. 해당 리스트를 돌며 n번째 주문의 순서보다 앞에있는 금지 주문이면 n에 1을 더해준다.

    3. 결과를 문자로 반환한다.



```python
def solution(n, bans) :
    spell_converted_list = []
    
    for ban in bans :
        spell_converted_list.append(convert_26(ban))
    spell_converted_list = sorted(spell_converted_list)
        
    for spell_converted in spell_converted_list :
        if spell_converted <= n :
            n += 1
                
    return convert_10(n)
```
