---
layout: single
title:  "Attention is all you need"
categories: Thesis
tag: [AI, blog, jekyll]
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


# Abstract


대표적인 sequence transduction(EX 문장 번역 등) 모델은 인코더와 디코더를 포함하는 RNN or CNN으로 구성



가장 뛰어난 성능을 보이는 모델은 ***attention mechanism***을 통해 인코더와 디코더를 연결



> 기존에 존재하던 Sequence Transduction 모델은 주요 연산을 RNN, CNN으로 수행, 보조적인 역할을 Attention 수행



저자가 제시하는 **Transformer**라는 새로운 네트워크 아키텍처는 RNN과 CNN을 완전히 배제하고 ***attention mechanism***만을 사용한다.



실험 결과 다음과 같은 우수한 성능을 보여줌



    - 영어-독일어 번역 작업에서 앙상블을 포함한 기존 최고의 성능 모델 보다 높은 성능 달성

    - 영어-프랑스어 작업에서 8개의 GPU로 학습 → 학습 비용 측면 및 성능 측면 모두 우수

    - 대규모 데이터 / 제한된 데이터 모두 영어 선거구 구문 분석에서 잘 적용됨 → Transformer가 다른 task에서도 일반화가 잘 됨


# Introduction


LSTM, GRU와 같은 RNN은 sequence 모델링 및 변환에서 SOTA



- RNN 한계



    - 입력 및 출력 시퀀스의 심볼 위치를 따라 계산

    - $$h_t = f(h_{t-1}, x_t)$$ 로 표현 → 이전 상태($$h_{t-1}$$)에 의존하여 순차적으로 처리

    - 이러한 순차적 처리로 병렬화가 불가능

    - 역전파를 위해 hidden state를 메모리에 추가로 저장해야 하므로 메모리 제약적



> 이를 위해 **인수분해 기법** 및 **조건부 계산**을 통해 효율을 증가시켰으나, sequence 연산 자체의 근본적 제약은 여전히 존재



- **Attention**



    - RNN만으로는 길이가 길어질수록 sequence 모델링 및 다음 은닉층으로 전달하는것이 힘듦

    - 입출력 sequence 거리와 관계없이 의존성을 모델링 가능

    - 대부분의 경우에서 RNN 모델과 함께 사용됨



- ***Transformer***



    - 순환 구조를 완전히 제거

    - **Attention**만으로 입출력 간 전역 의존성을 도출


## Background


순차적 계산을 줄이고 병렬화를 시도하려고 했던 CNN 모델들은 CNN을 기본 블록으로 모든 위치를 병렬 처리



→ 여전히 내부 은닉층이 깊어지면 거리에 따라 선형적 or 로그적으로 연산량이 증가함 → 먼 거리에 대한 의존성 학습이 어려움



- Transformer



    - 임의의 두 위치 간 관계를 상수 시간으로 계산

    - 거리에 무관한 연산량



    - Attention-weighted positions 평균화로 해상도 감소 → 연산 횟수 감소

    - Multi-Head Attention으로 상쇄 가능



- ***Self-Attention(Intra-Attention)***



    - 단일 sequence(문장) 내 서로 다른 위치를 연관시킴

    - 이전부터 Self-Attention은 독해, 추상적 요약, 텍스트 수반 등 다양한 작업에서 사용됨



    ◆ RNN, CNN을 사용하지 않고 self-attention만으로 입출력을 최초로 표현 → **Transformer**


# Model Architecture


전체적 구조로 **Encoder-Decoder** 사용



- ***Encoder*** : 입력 sequence ($$x_1, x_2, ..., x_n$$) → 연속 표현 ($$z_1, z_2, z_n$$)



    - 입력 텍스트를 토큰화 → 정수 ID로 매핑 → 실수 벡터 변환

    - 이를 통해 단어 각각의 의미적 유사성이 벡터 공간에서 거리로 표현됨



- ***Decoder*** : 연속 표현 ($z_1, z_2, z_n$) → 출력 sequence ($y_1, y_2, ..., y_n$) 생성


## Encoder and Decoder Stacks


- Encoder



    - 6개의 동일한 layer를 쌓음

    - 각 레이어는 2개의 sub-layer 존재

        1. Multi-head self-attention

        2. Position-wise fully connected feed-forward network

    - 이 각각의 sub-layer에 잔차 연결을 적용한 다음 layer 정규화를 적용

    - 출력은 $LayerNorm(x + Sublayer(x))$

    - 모델의 모든 sub-layer와 embbeding의 출력 차원은 512

    - Encoder에서는 문장의 전체적인 의미와 맥락을 파악



- Decoder



    - 6개의 동일한 layer를 쌓음

    - 각 레이어는 3개의 sub-layer 존재

        1. Masked multi-head self-attention (해당 layer를 수정하여 현재 위치와 이전 위치의 정보만을 사용 (미래 위치 차단))

        2. Multi-head attention over encoder output (Query는 이전 decoder layer, Key, Value는 encoder output)

        3. Position-wise feed-forward network

    - Encoder와 동일하게 잔차연결 + layer 정규화 적용

    - Decoder에서는 입력 문장에 대해 출력 문장을 하나씩 순서대로 생성


## Attention


- **Scaled Dot-Product Attention**



    - 입력 : Query, $d_k$(Key), $d_v$(Value)

    - 출력 : $Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

        - ${QK^T}$ : 모든 Query와 모든 Key의 내적 → 유사도 계산 → 각 단어들의 문법적, 의미적 연관 확인

        - $\left(1/\sqrt{d_k}\right)$ : Query와 Key의 차원이 커질수록 내적 값이 커지므로, Query와 Key의 차원을 제곱근으로 나눠서 정규화 (스케일링)

            

            ★ $\left(\frac{QK^T}{\sqrt{d_k}}\right)$ : Attention scores 계산



        - softmax : 각 단어 별 Attention socres를 확률 분포로 변환



        - $V$ : Value와 가중합을 취함으로써 가장 확률 분포로 변환된 Attention scores를 기반으로 확률에 비례하여 모든 Value를 가중 평균



    - 정리



    > Query, Key의 내적으로 유사도 계산 → 정규화를 통한 스케일링 → softmax를 통한 확률 분포 변환 → Value들의 가중 평균 계산



***


- **Multi-Head Attention**



    - Single-Head Attention의 경우 하나의 관점에서만 attention scores를 계산

    - 여러 개의 관점에서 attention scores를 계산하여 병합하여 사용

        1. 문법적 관계

        2. 의미적 유사성

        3. 감정 관계

        4. 개채명 인식

        5. ...



    - $MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) * W^O$

    - $head_i = Attention(Q * W^Q_i, K * W^K_i, V * W^V_i)$

        - 각 헤드의 가중치($W$)가 다름 → 서로 다른 관점에서 해석



    - 각 헤드는 Single-Head Attention보다 작은 차원으로 이루어져 있어 총 계산량은 거의 동일



***


- **Applications of Attention in Model**



    - ***Encoder_Decoder Attention***



        - Query : 이전 Decoder layer

        - Key, Value : Encoder Output



        - 결론 : Decoder에서 단어를 생성할 때마다 입력 sequence의 모든 위치를 참조하여 다음 단어를 생성



    - ***Encoder***



        - self-attention layers가 포함

        - Key, Value, Query 모두 이전 Encoder 계층의 출력에서 가져옴

        - 모든 위치를 참조하여 문맥 이해



    - ***Decoder***



        - self-attention layers가 포함

        - 미래의 정보가 과거로 흘러들어가는 것을 방지 → Masking


## Position-wise Feed-Forward Networks


Attention에는 sub-layers 이외에도 완전 연결 피드 포워드 네트워크도 존재



- **Fully connected feed-forward**



    - 두 개의 선형 변환과 사이에 ReLU 활성화 함수 적용

    - $FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$



    - 비선형 변환 → attention이 처리한 정보를 변환을 통해 더욱 복잡한 관계를 학습


## Embeddings and Softmax


> 학습된 임베딩을 사용하여 입력 및 출력 토큰을 같은 크기의 벡터로 통일



- 학습된 임베딩 : 특정 단어(벡터)가 처음에는 무작위 벡터 → 학습 과정에서 의미를 획득



    - 해당 과정은 학습 데이터에서 패턴을 발견하여 특정 벡터의 방향으로 학습



- Transformer에서 입출력 토큰을 임베딩



    - 입력 : 입력 문장에 대해 이해 및 의미 파악(학습)

    - 출력 : 다음에 나올 단어 예측을 위해 출력 토큰을 다시 임베딩하여 추론에 사용



        - 디코더의 출력 → 벡터 출력 → 선형 변환 → softmax → 다음 토큰 생성 확률 예측



> 입력 임베딩, 출력 임베딩, softmax 전 선형 변환 간 **동일한 가중치 행렬** 공유 / 임베딩 layer에서는 가중치에 벡터 크기의 제곱근을 곱해줌



- 동일한 가중치를 공유 → 의미적 일관성을 보장



- 벡터 크기의 제곱근을 곱해줌 → Positional encoding과 크기 균형을 맞춰주기 위해(스케일링)


## Positional Encoding


Transformer에는 순환 및 합성곱 존재 X → 순서 정보를 모름



> Encoder, Decoder의 입력 임베딩에 Positional Encoding 추가



- $PE(pos, 2i) = sin(pos/1000^{2i/d_{model}})$



- $PE(pos, 2i+1) = cos(pos/1000^{2i/d_{model}})$



- 모델이 **상대적 위치**를 학습



    - 각 차원마다 서로 다른 정현파를 가짐

        - 단어들 간 거리가 짧은 패턴 ~ 긴 패턴까지 학습 가능



> $PE(pos+k) = T_K \times PE(pos)$



- 단어 간 거리가 같다면 pos(위치)는 달라져도 $T_K$는 변하지 않음



    - $T_K$는 **거리 K**를 변환 행렬로 구현한 것 → 삼각함수의 덧셈정리로 구현



- 즉, pos 값과는 무관한 일관된 패턴 학습 가능



- 자연어 특성 상 절대적 위치는 다르지만, 상대적인 구조는 동일하기에 상대적 위치를 학습



- 상대적 구조를 표현하는 것이 $T_K$


# Why Self-Attention


RNN/CNN 대신 Self-Attention을 사용한 이유



- 비교 기준



    1. layer당 총 계산 복잡도 

    2. 병렬화할 수 있는 계산의 양

    3. 네트워크 장거리 의존성 경로 길이



- n : sequence length

- d : embedding dimension

- k : kernel size

- r : restricted attention span



***

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |

|------------|---------------------|---------------------|-------------------|

| Self-Attention | $O(n² · d)$ | $O(1)$ | $O(n)$ |

| Recurrent | $O(n · d²)$ | $O(n)$ | $O(n)$ |

| Convolutional | $O(k · n · d²)$ | $O(1)$ | $O(log_k(n))$ |

| Self-Attention (restricted) | $O(r · n · d)$ | $O(1)$ | $O(n/r)$ |



***


- ***계산 복잡도***



    - 대부분의 경우 임베딩 차원이 sequence 길이 보다 큼

        - 임베딩 차원이 클수록 표현력이 증가

    - 따라서 임베딩 차원이 선형적으로 증가하는 Self-Attention의 계산 복잡도가 CNN,RNN보다 효율적



- ***병렬 처리 능력***



    - RNN : 순차적 계산이 필수적 → $O(n)$

    - CNN : 병렬 처리 가능

    - Self-Attention : 병렬 처리 가능



- ***장거리 의존성***



    - RNN : 순차적으로 계산되어 전달되므로 $O(n)$

    - 단일 CNN : sequence 길이 n에 대하여 k개의 커널이 필요함 → $O(n/k)$

    - 확장 CNN : 간격을 넓혀서 더 넓은 범위를 연결 → $O(log_k(n))$

    - Self-Attention : 모든 위치 쌍이 직접 연결되어 있으므로 $O(1)$



- 매우 긴 sequence를 처리할 때 계산 성능 향상을 위해 출력 위치 중심으로 입력 sequence에서 $r$만큼으로 이웃 제한 → 장거리 의존성이 $O(n/r)$로 증가



***

