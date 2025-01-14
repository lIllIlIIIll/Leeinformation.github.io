---
layout: single
title:  "프롬프트 엔지니어링 - 4. Examples 및 여러가지 Shot 프롬프트"
categories: AI
tag: [python, Machine Running, Prompt Engineering]
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


> # 4. Examples and Shot Prompting


**zero-shot, one-shot, few-shot** 프롬프트와 같이 예시를 사용하는 잘 알려진 프롬프트 기법이 많다.



각 접근 방식과 효과적인 예시 설계의 원칙을 이해하면 Solar 모델의 성능을 크게 향상시킬 수 있다.



***


>> ## 4.1 Zero-Shot Prompting


>>> 4.1.1 기본 개요


**Zero-Shot prompting**은 모델에게 사전 예시 없이 작업을 수행하도록 요청하는 기법이다.



**Zero-Shot** 프롬프트를 생성하는 방법은 여러가지가 존재한다.



이는 언어 모델의 강력한 기능 중 하나로, 최소한의 정보로 다양한 작업을 수행할 수 있게 해준다.



***


>>> ### 4.1.2 예시


```python

message = [

    {

        "role":"user",

        "content": "Classify the following text as positive, negative, or neutral. Text: I thought the macaron flavor was just okay. Sentiment: { }"

    }

]



response = get_completion(messages=message)

print(response, "\n\n")

```



단순하게 어떤 문장이 주어졌을 때 그 문장이 긍정인지, 중립인지, 부정인지를 판별하는 것을 Zero-Shot prompting의 예시 중 하나라고 볼 수 있다.



***


>> ## 4.2 One-Shot Prompting


>>> ### 4.2.1 기본 개요


**One-Shot prompting**은 모델에게 작업을 완료하도록 요청하기 전 하나의 예시만 제공하는 기법이다.



***


>>> ### 4.2.2 예시


```python

message = [

    {

        "role":"user",

        "content": "Translate the following sentence into French: I like reading books."

    },

    {

        "role":"assistant",

        "content": "J'aime lire des livres."

    },

    {

        "role":"user",

        "content": "Translate the following sentence into French: I love studying language"

    },

]



response = get_completion(messages=message)

print(response, "\n\n")

```



모델이 해야하는 작업이 "**I love studying language**"를 불어로 번역하기 일 때, 이전에 "**I like reading books**"를 번역하면 어떻게 되는지 먼저 하나의 예시를 제공하고 이후 작업을 수행한다.



***


>> ## 4.3 Few-Shot Prompting


>>> ### 4.3.1 기본 개요


**Few-Shot Prompting**은 모델에 작업을 완료하도록 요청하기 전 일반적으로 2 ~ 5개의 예시를 제공하는 기법이다.



이 접근 방식은 모델에 원하는 출력 유형에 대한 여러 참조를 제공하여 더 많은 컨텍스트를 제공하고 출력을 더 정확하거나 특정 패턴에 맞게 만들 수 있다.



> **Few-Shot learning**과 **Few-Shot Prompting**은 서로 다른 개념이다.



> **Few-Shot learning**은 몇 개의 예시만으로 모델 파라미터를 조정하는데 초점을 맞춘 광범위한 머신 러닝 접근 방식이다.



> **Few-Shot Prompting**은 모델 매개변수가 변경되지 않는 생성형 AI의 프롬프트 디자인에 특별히 적용된다.



***


>>> ### 4.3.2 예시


```python

message = [

    {

        "role": "user",

        "content": "2+10:"

    },

    {

        "role": "assistant",

        "content": "twelve"

    },

    {

        "role": "user",

        "content": "4+52:"

    },

    {

        "role": "assistant",

        "content": "fifty-six"

    },

    {

        "role": "user",

        "content": "100+301:"

    }

]



response = get_completion(messages=message)

print(response, "\n\n")

```


덧셈에 대한 예시 여러 개(5개 이하)를 준 다음 원하는 요청인 100+301을 수행한다.



***


>> ## 4.4 Solar 사용 시 효과적인 예시 디자인을 위한 방법


**Exemplar Quantity**

- 대부분이 프롬프트의 예시 수가 많을 수록 모델 성능이 향상된다.



- 예시를 제공하면 예시에서 사용된 형식으로 모델이 응답한다. 문장 구조가 복잡하고 다양한 경우에도 많은 예시를 사용하여 모델을 안내하면 작업을 잘 수행하는데 도움이 된다.



**Exemplar Similarity**

- 작업과 유사한 예시를 선택한다. 



- 예를 들어, 뉴스 기사를 요약하는 경우 동일한 형식의 예시를 사용하면 원하는 최상의 결과를 얻을 수 있다.



**Exemplar Format**

- 공통 템플릿을 사용한다. 최적의 형식은 작업마다 다를 수 있다.



- 학습 데이터에서 자주 발견되는 형식이 성능 향상으로 이어지는 경향이 있다.



- 비정상적인 형식의 기호나 마크를 사용하면 결과가 생성될때마다 안정성이 떨어진다.



***

