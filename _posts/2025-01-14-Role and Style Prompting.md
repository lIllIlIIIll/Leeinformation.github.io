---
layout: single
title:  "프롬프트 엔지니어링 - 5. Role 및 Style 프롬프트"
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


> # 5. Role and Style Prompting


이번 포스팅에서는 프롬프트 디자인을 위한 두 가지 효과적인 기법을 살펴본다.



**Role 프롬프트** 그리고 **Style 프롬프트**, 두 기법 모두 Solar 모델에 적합하다.



***


>> ## 5.1 모델에 역할 할당


>>> ### 5.1.1 Role Prompting


**persona prompting**이라고도 하는 이 기법은 프롬프트 내 대규모 언어 모델에 특정 역할을 할당한다.



이 접근 방식은 개방형 작업에 대해 보다 바람직한 결과를 도출할 수 있고 경우에 따라 벤치마크의 정확도를 높일 수 있다.



두 가지의 유형이 존재하는데 각각 다음과 같다.



(1) 화자 별 프롬프트 : LLM에 역할을 할당하는 프롬프트



(2) 대상 별 프롬프트 : 대화 대상을 지정하는 프롬프트



***


**(1) 화자 별 프롬프트**



```python

message = [

    {

    "role": "user",

    "content": """You are an expert lawyer.

Please interpret the case regarding the Good Samaritan law from the following. 

case: 

The California Supreme Court recently made an interesting ruling related to the "Good Samaritan law." This ruling concerns the legal dispute between Alexandra Van Horn and Lisa Toti. Here’s what happened: In 2004, Alexandra Van Horn and Lisa Toti were heading to a banquet with colleagues from the same factory. However, the car that Van Horn was in got into an accident, and Lisa Toti, who was in another car, pulled Van Horn out of her vehicle, which was stuck at the accident scene. Unfortunately, during this rescue, Van Horn sustained injuries that left her unable to use her lower body. Subsequently, Van Horn filed a lawsuit against Toti, claiming that she was paralyzed due to Toti's actions while trying to rescue her. The California Supreme Court ruled in a 4-3 decision, acknowledging Toti's responsibility."""

    }

]

response = get_completion(messages=message)

print(response, "\n\n")

```



위와 같이 본문을 설명하기 전 모델에게 역할을 할당해 준다.



해당 예시에서는 **변호사**로 역할을 할당해 주었다.



그렇다면 모델은 변호사로써 해당 사례를 전문적으로 해석할 것이다.



***


(2) 대상 별 프롬프트



```python

message = [

    {

        "role": "user",

        "content": """You are currently talking with an elementary school student.

Please interpret the case regarding the Good Samaritan law from the following.

case: 

The California Supreme Court recently made an interesting ruling related to the "Good Samaritan law." This ruling concerns the legal dispute between Alexandra Van Horn and Lisa Toti. Here’s what happened: In 2004, Alexandra Van Horn and Lisa Toti were heading to a banquet with colleagues from the same factory. However, the car that Van Horn was in got into an accident, and Lisa Toti, who was in another car, pulled Van Horn out of her vehicle, which was stuck at the accident scene. Unfortunately, during this rescue, Van Horn sustained injuries that left her unable to use her lower body. Subsequently, Van Horn filed a lawsuit against Toti, claiming that she was paralyzed due to Toti's actions while trying to rescue her. The California Supreme Court ruled in a 4-3 decision, acknowledging Toti's responsibility."""

    }

]



response = get_completion(messages=message)

print(response, "\n\n")

```



마찬가지로 본문을 설명하기 전 모델에게 대화 대상이 누구인지를 알려준다.



해당 예시에서는 초등학생과 대화하고 있다고 지정하였다.



그렇다면 모델은 어린아이도 이해할 수 있도록 해당 사례를 설명해 줄 것이다.



***


>>> ### 5.1.2 Style Prompting


이 기법에는 프롬프트 내에서 원하는 스타일, 어조 또는 장르를 정의하여 대규모 언어 모델의 출력에 영향을 주는 것이 수반된다.



Role prompting을 통해 비슷한 결과를 얻을 수 있는 경우가 많다.



다음의 두 프롬프트를 보면 명백한 차이가 존재한다.



***


```python

message = [

    {

        "role": "user",

        "content": "Write a ten-word sentence about BTS, the Korean singers, in a humorous tone."

    }

]



response = get_completion(messages=message)

print(response, "\n\n")

```



***



```python

message = [

    {

        "role": "user",

        "content": "Write a ten-word sentence about BTS, the Korean singers."

    }

]



response = get_completion(messages=message)

print(response, "\n\n")

```



***


**문체 제약 조건**

- Style prompting의 경우 모든 언어에 존재하는 문체 제약 조건을 사용할 수 있다. 이는 종종 예시와 같이 문서 유형 앞에 형용사를 배치하는 것으로 구성된다.



    - "상사에게 ***공식적인*** 이메일 쓰기"

    - "***재미있는*** 픽업 라인 작성"



다음은 몇 가지 문체 제약의 예시이다.

- 글쓰기 스타일 : 기능적, 꽃말, 솔직, 산문, 화려, 시적

- 어조 : 극적, 유머러스, 슬픈, 낙관적, 공식적, 비공식적, 독단적, 공격적

- 기분 : 화, 두려움, 행복, 슬픔

- 속도 : 빠르고 느린 속도



***


>> ## 5.2 일관성 있는 응답 유지


**제약 조건**

- 예상되는 응답 형식에 대한 명확한 가이드라인을 정의한다.

- 프롬프트 전체에 일관된 용어를 사용하여 혼동을 피한다.



제약 조건은 특히 Zero-Shot 프롬프트 엔지니어링 기법에서 효과적이다.



***


>>> ### 5.2.1 예상 응답 형식에 대한 명확한 가이드라인 정의


예상 출력의 구조와 구성 요소를 지정한다.



- 목록 형식 : "규칙적인 운동의 다섯 가지 이점 목록을 제공해줘"

- 단락 형식 : "재생 에너지의 중요성에 대해 한 단락으로 설명해줘"

- 대화 형식 : "두 명의 인물이 주말 계획을 논의하는 대화를 작성해줘. 단, 세 차례에 걸친 대화 교환을 포함해야함."



***


>>> ### 5.2.2 프롬프트 전체에서 일관된 용어 사용


모든 관련 프롬프트에서 일관되게 사용할 특정 용어와 문구를 선택한다.



이는 혼란을 최소화하고 모델이 문맥을 더 잘 이해하는데 도움이 된다.



예를 들어, 모델과 상호작용하는 사람을 지칭하기 위해 "사용자"를 선택하는 경우 "클라이언트" or "참가자"로 바꾸는 대신 "사용자"를 사용한다.


>>> ### 5.2.3 예시


다음의 텍스트를 읽고 요약한다. 길이를 제한하고, 비슷한 개념을 함께 그룹화하고, 유형별로 분류할 것.



**Text** :

> 액면가 : 피보험자 사망 시 보험금 수령자에게 지급되는 달러 금액이다.. 여기에는 배당금으로 구매한 보험이나 보험 부수 보험에서 지급될 수 있는 기타 금액은 포함되지 않는다.



> 재정 보증 보험 : 보험 청구인, 의무자 또는 피보험자에게 재정적 손실의 발생이 증명되면 손실이 지급되는 보증 채권, 보험 증권 또는 보험사가 발행한 경우 배상 계약 및 상기 유형과 유사한 모든 보증을 말한다.



> 화재 보험 : 화재로 인한 건물 및/또는 내용물의 손실 또는 손상에 대한 보상을 제공한다.



> 좋은 운전자 할인 : 좋은 운전자 할인을 받으려면 보험에 가입된 차량의 모든 운전자는 면허를 취득한 지 3년 이상이고 운전 기록에 벌점이 1점 이하이어야 하며 타인의 신체 상해 또는 사망을 초래한 사고에서 “과실”이 없는 것으로 판정받지 않아야 한다.



> 유예 기간 : 보험료 납부 마감일 직후에 보험료를 납부하여 보험을 중단 없이 계속 유지할 수 있는 지정된 기간이다. 이는 생명 및 건강 보험에만 적용된다. 유예 기간이 제공되는지 여부와 허용되는 기간(있는 경우 며칠인지)은 보험 약관을 확인 바람.



> 보장된 보험금 : 보험 가입자가 보험 가입 가능성에 대한 증거 없이도 미래의 지정된 시기에 명시된 금액의 생명 보험에 추가로 가입할 수 있는 옵션이다.



> 건강 보험 : 의료비 또는 치료에 대해 명시된 금액을 지급하는 보험입니다. 건강 보험은 다양한 옵션을 제공할 수 있으며 보장 방식도 다양하다.



> 주택 소유자 보험 : 주택 소유에 따른 위험에 대한 보장을 선택적으로 조합한 보험입니다. 화재, 도난, 기물 파손, 지진 및 기타 위험으로 인한 손실이 포함될 수 있다.



> 분쟁 불가 조항 : 계약이 발효된 후 일정 기간(보통 2년) 동안 계약의 유효성에 대해 이의를 제기하지 않기로 회사가 동의하는 보험 약관 조항이다.



***


위의 텍스트에 따른 모델의 요약 기댓값은 다음과 같다.



**Output** :

1. 보험 유형 :

    - 재정 보증 보험 : 재정적 손실을 보장하는 계약입니다.

    - 화재 보험 : 화재로 인한 손실 또는 손상을 보상합니다.

    - 건강 보험 : 의료비 또는 치료비를 지급합니다.

    - 주택 소유자 보험 : 화재, 도난 등 주택 소유로 인한 위험을 보장합니다.



2. 정책 특징 :

    - 액면가 : 피호범자 사망 시 보험금 수령자에게 지급되는 금액입니다.

    - 좋은 운전자 할인 : 운전 기록이 깨끗한 운전자에 대한 할인입니다.

    - 유예 기간 : 보험료 만기일 이후 보험을 중단 없이 계속 유지할 수 있는 기간입니다.

    - 보장된 보험 : 보험 가입 가능성에 대한 증거 없이 추가 보험에 가입할 수 있는 옵션입니다.

    - 분쟁 불가 조항 : 일정 기간이 지나면 회사가 계약의 유효성에 대해 이의를 제기할 수 없도록 하는 조항입니다.



***



위와 같이 사용자가 준 텍스트를 조건에 맞게 요약할 수 있다.



이는 예상 응답 형식에 대한 명확한 가이드라인(길이 제한, 그룹화, 유형별 분류)을 제공함으로써 원하는 결과를 얻도록 유도할 수 있다.

