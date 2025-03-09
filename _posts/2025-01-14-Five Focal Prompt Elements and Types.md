---
layout: single
title:  "프롬프트 엔지니어링 - 3. 5가지 프롬프트 핵심 요소와 유형"
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


> # 3. Five Focal Prompt Elements and Types


이번 포스팅에서는 잘 구성된 프롬프트의 다섯 가지 핵심 요소와 이러한 요소를 활용하는 다양한 유형의 프롬프트에 대해 살펴본다.



다섯가지 요소는 **Instruction, Context, Example, Input Data, Output**이다.



이러한 요소들은 Solar 모델과 효과적인 커뮤니케이션의 토대가 되며, 생성된 응답이 원하는 결과와 일치하도록 보장한다.



***


>> ## 3.1 Focal Elements


**Instruction**



- **Instruction**은 모델에게 수행할 작업을 지시하는 프롬프트의 핵심 구성 요소이다. 좋은 **Instruction**은 명시적이고 간결하며 목표 지향적이어야 한다.



***



**Context**



- **Context**는 모델이 보다 관련성 높은 응답을 생성하는데 도움이 되는 배경 정보 또는 프레임을 제공한다. 여기에는 과거 상호작용, 사용자별 정보 또는 작업 관련 세부 정보가 포함될 수 있다.



***



**Example**



- **Example**은 모델의 응답 스타일과 형식을 안내하는데 유용하다. 하나 이상의 예제를 제공하면 모델이 어조, 내용 또는 출력 구조 측면에서 예상되는 것을 이해하는데 도움이 될 수 있다.



***



**Input Data**



- **Input Data**는 텍스트 구절, 숫자 데이터 또는 기타 유형의 입력 등 모델이 작업해야 하는 특정 콘텐츠를 의미한다. 이 요소를 통해 모델은 제공된 콘텐츠를 기반으로 상세한 응답을 생성할 수 있다.



***



**Output**



- **Output**은 응답의 구조, 톤 또는 길이 등 응답의 모양을 지정한다. 원하는 출력을 명확하게 정의하면 모델이 사용자의 기대에 맞게 응답을 조정하는 데 도움이 될 수 있다.



***



이 다섯가지 핵심 요소를 서로 다른 방식으로 결합하면 다양한 유형의 프롬프트가 생성된다.


다음은 네 가지 프롬프트 유형이다.



- **Type A** : Instruction + Output



- **Type B** : Instruction + Context + Output



- **Type C** : Instruction + Context + Example + Output



- **Type D** : Instruction + Input Data + Output



위의 유형들을 사용하여 프롬프트를 만들면 얻을 수 있는 이점이 존재한다.



1. 각 작업에 최적의 조합 찾기

    - 모든 작업에는 각기 다른 유형의 프롬프트 구조가 필요하다. 이러한 유형을 테스트하여 각 작업에 가장 효과적으로 작동하는 최적의 조합을 찾을 수 있다.



2. 프롬프트 최적화 테스트의 용이성

    - 이러한 유형을 사용하면 프롬프트를 더 쉽게 테스트하고 최적화할 수 있기에 모델의 응답을 효율적으로 개선하고 세분화할 수 있다.



    ***

