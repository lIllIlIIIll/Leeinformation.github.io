---
layout: single
title:  "프롬프트 엔지니어링 - 1. 기본 프롬프트 구조"
categories: AI
tag: [python, Machine Learning, Prompt Engineering]
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


> # 1. Basic Prompt Structure


Upstage에서 개발한 **Solar** 모델을 프롬프트 엔지니어링하는 기본적인 틀 및 심화 과정에 대해 요약한다.



**프롬프트 엔지니어링**은 해당 모델 뿐만 아니라 다른 LLM에서 사용될 수 있다.



해당 요약글은 다음의 주소를 기반으로 정리하였음



▶ [프롬프트 엔지니어링](https://github.com/UpstageAI/solar-prompt-cookbook) ◀



***


**프롬프트 엔지니어링** : LLM(대규모 언어 모델)으로부터 원하는 결과를 얻기 위해 **프롬프트**를 설계하는 과정



프롬프트의 내용, 형식, 구조, 단어 선택 등에 따라 모델의 성능이 좌지우지됨.



***


>> ## 1.1 파라미터의 이해


프롬프트 엔지니어링을 수행하는 과정에서 파라미터는 모델의 작동 방식 및 출력을 제어하는 핵심적인 역할이다.



다음은 파라미터에 대한 설명이다.


>>> ### 1.1.1 파라미터의 이해 - 파라미터


**Model**



- 상호 작용을 하려는 특정 모델


**Max_Tokens** :



- 해당 파라미터는 출력에서 전체 토큰의 수(단어 or 단어 일부)를 제한한다. **Max_Tokens**를 제어하는 것은 모델 출력의 최대 길이를 설정할 수 있는데, 이는 과도하게 긴 응답을 피하거나 API 사용 비용 제어, 특정 유스케이스에 맞게 출력을 제어하는 데 유용하다.



- **Hard Stop** :

  - 모델이 특정 한도를 초과하여 토큰을 생성하지 못하도록 방지

  - 토큰 한도에 도달하면 단어 or 문장 중간에서 생성이 중지될 수 있다.



- **Prompt Tokens** :

  - 입력 프롬프트에서 토큰의 수



- 만약 **Max_Tokens**이 설정되면, 입력 토큰과 최대 토큰의 합은 모델의 문장 길이보다 작거나 같아야 한다.(≤ 4096)


**Temperature** :



- 해당 파라미터는 모델 응답의 무작위성 or 창의성을 제어한다.

  - 값이 높을수록 유연성이 높아져 더 다양한 텍스트를 생성

  - 값이 낮을수록 모델이 더 결정론적 → 더 정확하고 일관된 출력을 생성



- 유효 범위는 0과 2.0 사이

  - 0.0 : 출력은 결정론적이고 예측 가능 → 모델이 매번 동일한 프롬프트에 대해 동일한 응답을 반환할 가능성이 높음

  - 0.7 : 모델이 창의적이면서 여전히 집중하는 균형 잡힌 수준 → 답변은 다양할 수 있으나 주제에 집중하는 경향

  - 2.0 : 매우 창의적이거나 무작위적 결과를 장려 →잠재적으로 더 특이 or 다양한 반응을 생성


**Top_P** :



- 토큰 선택의 누적 확률을 고려하여 모델 결과의 무작위성을 제어하는 방법. **Top_P**는 모델이 응답을 생성할 때 얼마나 안전 or 위험한지 제어할 수 있다. 값이 낮으면 모델의 샘플링 범위가 줄어들어 확률이 높은 토큰을 고수하고, 값이 높으면 응답의 다양성이 증가



- **Top_P = 0.9** : 모델은 누적 확률이 90%인 가장 작은 집합에서 토큰을 샘플링


◆ **Temperature** vs **Top_P**



**Temperature** : 모델이 전반적으로 얼마나 창의적인지에 영향



**Top_P** : 최종 응답에서 얼마나 많은 높은 확률 토큰을 고려하는지에 영향


>>> ### 1.1.2 파라미터의 이해 - 예시


```python

config_model = {

    "model": "solar-pro",

    "max_tokens": 2000,

    "temperature": 0.7,

    "top_p": 0.9,

}



config_robust = {

    "model": "solar-pro",

    "messages": [

        {

            "role": "user",

            "content": "What are the potential benefits of AI in healthcare?"

        }

    ],

    "max_tokens": 2000,

    "temperature": 0.0,

    "top_p": 0.8

}



response = client.chat.completions.create(**config_robust)

print(response.choices[0].message.content, '\n\n')

```



***


>> ## 1.2 구조 이해


>>> ### 1.2.1 구조이해 - 입력 구조


**messages** :



- 대화 컨텍스트가 포함된 배열. 사용자와 모델 간 교환 내용이 포함된다.



- **'role'** :

  - **role**은 메시지의 출처를 나타내는 **'user'**, **'assistant'** 또는 **'system'**

  - **role : 'system'**인 경우 어시스턴트의 행동, 어조 및 지식 기반을 설정하여 초기 지침 역할

  - **role : 'user'**인 경우 사용자로의 메시지임을 지정

  - **role : 'assistant'**인 경우 사용자의 쿼리를 해결하거나 대화를 이어나가기 위해 AI가 생성한 응답 포함


>>> ### 1.2.2 구조이해 - 예시


```python

{

  "role": "system",

  "content": "You are my Assistant. Your role is to answer my questions faithfully and in detail."

}



{

  "role": "user",

  "content": "Hello, Solar. Can you help me plan a weekend trip to New York City?"

}



{

  "role": "assistant",

  "content": "Hello! I'd be happy to help you plan your weekend trip to New York City. Let's start by discussing your interests and preferences. Are you looking for sightseeing, shopping, diningor perhaps a mix of all?"

}

```



***


>> ## 1.3 시스템 프롬프트 이해


>>> ### 1.3.1 시스템 프롬프트


**시스템 프롬프트**는 AI 모델이 사용자 입력을 해석하고 이에 응답하는 방식을 형성하는데 중요한 역할을 한다.



프롬프트 엔지니어링의 맥락에서 시스템 프롬프트를 효과적으로 이해하고 활용하면 모델의 동작을 안내하고 사용자의 기대에 부합하는 응답을 보장하는데 도움이 될 수 있다.



◆ 시스템 프롬프트가 짧으면 응답이 짧고, 시스템 프롬프트가 길면 일반적으로 응답이 길어지는 경향이 있다.



***

