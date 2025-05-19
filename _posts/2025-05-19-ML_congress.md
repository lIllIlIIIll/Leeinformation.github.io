---
layout: single
title:  "[AI 부트캠프] ML 경진대회 - 아파트 실거래가 예측"
categories: Bootcamp
tag: [패스트캠퍼스, 패스트캠퍼스AI부트캠프, 업스테이지패스트캠퍼스, UpstageAILab, 국비지원, 패스트캠퍼스업스테이지에이아이랩, 패스트캠퍼스업스테이지부트캠프]
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


# 1. 경진대회 개요


***

5월이 시작되고, 서울의 아파트 실거래가를 예측하는 경진대회가 시작되었다.



첫 회의 때 서로가 알고있던 도메인을 공유하여 주어진 데이터 내에서 feature들의 중요도를 파악하고, 추가적으로 어떤 데이터를 추가하여 모델의 성능을 향상시킬 수 있는지에 대해 논의하였다.



이전 여러 경진대회에서 경험했던 바와 같이 물론 직접 훈련에 들어갔을 때 우리가 중요하게 생각했던 feature가 필요성이 떨어질 수 있고, 반대의 경우도 가능했기에 전체적으로 어떻게 경진대회를 진행할지에 대해서 논의하였다.

***


# 2. EDA


***



머신러닝 파이프라인에 있어 가장 어렵고 중요한 부분이라고 생각하는 파트이다.



주어진 데이터를 시각화 or 통계적으로 분석하여 변수들간의 상관관계를 파악하고, 파생 변수 생성 및 필요 없는 변수 제거 등을 수행하는데 근거가 되기에 각 feature가 어떤 분포를 가지는지 확인하는데 집중했다.



데이터를 전체적으로 살펴보았을 때, 굉장히 결측치가 많았다.




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/train_missing.PNG?raw=true)




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/test_missing.PNG?raw=true)




전체 52개의 feature 중 5~6개를 제외하면 70% 이상의 결측치를 가졌고, 심지어 각 feature를 target과 비교하였을 때 이상치로 구분되는 값도 많았다.



또한, train 데이터와 test 데이터의 분포가 너무 달랐다.



    - train 데이터 : 근 30년간의 아파트 거래가격을 나타냄



    - test 데이터 : 2023년의 아파트 거래가격을 나타냄



> 이는 물가, 경제 상황 등의 외부적 요인에 따라 **target(금액)**의 가치가 달랐다.

>

> 검증셋을 2023년 데이터만으로 구성하는 방법을 사용 (물론 나중 모델 훈련시에는 전체를 검증셋으로 사용하여 RMSE를 낮추는 방향으로 진행하는 것이 더 좋았다.)



***



위에서 기술하였듯이 결측치가 많았기에 52개의 feature를 전부 사용하는 것은 모델 훈련에 있어 성능을 저하시킨다고 판단하였고, 실제로 데이터를 전처리 시키기 전, 정말 필요없다고 생각하는 feature들을 제거하였다.



그리하여 52개의 feature 중 남긴 feature는 다음과 같았다.



> 전용면적(㎡), 건축년도, 해제사유발생일, k-연면적, k-전용면적별세대현황(60㎡~85㎡이하),k-전용면적별세대현황(85㎡~135㎡이하), k-전체세대수, 주차대수, 계약(연), 좌표X, 좌표Y, 아파트명, 도로명, 등기신청일자, k-복도유형, k-단지분류(아파트,주상복합등등)

***


# 3. Preprocessing


***



앞서 진행하였던 **EDA**를 기반으로 전처리를 진행하였다.



- train 데이터



    - 중요도가 떨어진다고 생각하는 특성 Drop



    - 문자열 columns을 찾아서 좌우 공백 제거(같은 데이터인데 다르게 해석할 가능성 제거)



    - 데이터 중 중복으로 같은 값을 나타내는 feature Drop (번지, 본번, 부번)



    - **계약년월** → **계약(연), 계약(월)** 로 분해



    - **해제사유발생일** → **해제사유발생여부** 로 변환



    - **등기신청일자** → **등기신청여부** 로 변환



    - **세대당 주차대수** 특성 생성 **(주차대수 / 전체 새대수)**



    - 좌표값을 이용한 **군집화**



***


***

- bus, subway 및 외부  데이터



    - 좌표값을 이용하여 아파트와 각 버스정류장, 지하철역까지의 거리 계산



    - 한국은행 기준금리 추가



***


***

전처리 과정에서 가장 신경을 쏟았던 부분인 군집화 부분을 살펴보면...



1. 좌표의 결측치 처리



    ※ 데이터 내 **도로명**을 이용하여 카카오 API로 좌표를 추출 



    ※ 그래도 남아있던 결측치(재개발 등의 이유로)는 이상치로 간주하고 제거



2. Kmeans Clustering



    ※ **Kmeans Clustering**을 이용하여 좌표별 군집화 수행



    ※ 첫 군집화 시 다음과 같이 추출됨



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/first_cluster.PNG?raw=true)



        → 이상치로 보이는 것 들이 존재하기에, 이를 제거하고 다시 군집화



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/second_cluster.PNG?raw=true)



3. feature 생성



    ※ 군집화 모델로 추출한 **군집**을 **cluster**라는 새로운 feature로 생성

***


***



이러한 전처리 과정들을 거친 데이터는 다음과 같다.




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/df_head.PNG?raw=true)




***


# 4. LGBM Training


***



처음 훈련시킬 모델로는 SOTA 모델인 **LightGBM Regressor**를 사용하였다.



모델을 훈련할 때는 **Baysian Optimization**을 사용하여 다음의 탐색공간 내에서 최적의 파라미터를 찾도록 하였다.



```python

param_space = {

    'n_estimators': scope.int(hp.quniform('n_estimators', 300, 3000, 10)),

    'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),

    'num_leaves' : scope.int(hp.quniform('num_leaves', 2, 50, 1)),

    'max_depth': scope.int(hp.quniform('max_depth', 0, 40, 1)),

    'min_data_in_leaf' : scope.int(hp.quniform('min_data_in_leaf', 0, 50, 1)),

    'feature_fraction_bynode' : hp.uniform('feature_fraction_bynode', 0.001, 1.0),

    'bagging_fraction' : hp.uniform('bagging_fraction', 0.001, 1.0),

    'bagging_freq' : scope.int(hp.quniform('bagging_freq', 0, 30, 1)),

    'min_child_weight': hp.uniform('min_child_weight', 0, 10),

    'reg_alpha': hp.uniform('reg_alpha', 0, 1),

    'reg_lambda': hp.uniform('reg_lambda', 0, 1),

    'drop_rate' : hp.uniform('drop_rate', 0, 1)

}

```



***


***



최적의 하이퍼파라미터를 찾고, 최적의 모델로 검증셋에 대한 예측을 수행하였을 때, 다음과 같은 **RMSE**가 나왔다.




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/LGBM_result.PNG?raw=true)




또한 첫 submission을 제출하였고 다음과 같은 public score를 기록하였다.




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/first_sub.PNG?raw=true)




***


# 5. AutoML - Autogluon


***



LGBM으로 모델을 선정하고 데이터 전처리 및 모델 학습을 계속해서 하고 있을 때 문득 생각이 난 패키지였다.



예전에 데이콘에서 주최했던 대회에서 사용 경험이 있었기에 AutoML을 사용해보는 것도 좋을 것 같았다.



그리고 바로 **Autogluon**을 사용해 보았다.



먼저 훈련 당시 validation set에 대한 RMSE는 다음과 같이 나왔다.




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/AutoML_RMSE.PNG?raw=true)





확실히 낮아진 것을 볼 수 있었다. (스스로 KFold Cross Validation을 사용하여 validation set을 나누었다.)





다음으로 submission을 제출하여 Public RMSE를 확인해보았다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/AutoML.PNG?raw=true)




바로 LGBM을 사용한 것 보다 Public RMSE가 4000이나 낮게 나온다...



심지어 Autogluon의 파라미터를 지정할 때 잘못 지정해버려서 모든 feature를 사용하는 모델을 학습시켜버렸었다.



그런데 저 정도의 퍼포먼스를 보여줬고, 바로 모델을 LGBM에서 Autogluon으로 교체하였다.



***


## 5.1 추가 전처리


***



사실 AutoML 패키지에 포함된 모델들은 별 다른 전처리를 필요로 하지 않는다.



추후 알게된 내용인데, 내부적으로 데이터 전처리를 자동으로 해주고, 모델 훈련과정에서 스스로 적합한 파생 변수를 추가해주며, 검증셋도 나눠주기 때문이다.



하지만 우리팀은 '여기에 추가로 전처리를 해준다면 모델의 성능이 더 좋아지지 않을까?' 라고 접근하였다.



- LGBM에서 사용하였던 feature engineering 적용



- AutoML 내 TimeSeries를 이용하여 시계열 데이터로 분석



- 적용했던 feature engineering에서 여러 조합들 시도



AutoML을 사용하기로 결정한 이후부터 대회 마감까지 굉장히 여러가지 시도들을 많이 해보았었다.



***


# 6. 결과


결론적으로는 Raw Data 상태 그대로 AutoML을 사용하는 것이 성능이 가장 뛰어났다.



AutoML Documents를 살펴보니, 위에서 기술했듯이 내부적으로 스스로 feature를 생성하는 등의 전처리를 하기에 그랬던 것 같았다.



사실 이전 데이콘 경진대회에서는 데이터를 사람이 직접 어느정도 정제해주어야 성능이 좋아졌었고, 그에 맞게 이번 대회에서도 그러한 방식을 사용하였던 것이었는데, 도무지 성능이 오를 생각을 하지 않았었다.



결국 팀원이 Raw Data로 훈련시켰을 때 Public이 가장 낮았던 모델 하나와 Public은 다소 낮으나 전처리가 어느정도 이루어진 모델 하나를 제출하였다.




![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/대회결과.PNG?raw=true)





3등이라는 좋은 결과를 달성할 수 있었다.



실제로 Private 결과가 더 좋았던 모델이 있었기에 더 아쉬웠었다.


# 7. 회고


대회 결과로써는 팀원들 모두 만족하는 결과였다.



마지막으로 Public 등수는 4등이었고, Private 등수는 3등으로 올랐다.



결과적으로는 좋았으나, 팀원들 간 협업 부분에 있어서는 부족했던 것 같다. 이부분을 정리해보면...



- 잘했던 점



    > 팀원들 간 소통에 있어서는 서로 진행했던 전처리, 훈련 등을 구두적으로 공유하는 것은 순조롭게 진행되었다.



- 부족했던 점



    > 소통은 잘 되었으나 내용적으로 Github에 공유하는 부분이 부족했다. 즉, 코드 공유는 잘 안되었다.



나에게는 이런 머신러닝 대회를 팀으로 나가는 건 처음이었고 팀장을 맡아 진행하는 것도 처음이었다.



이후 MLOps 관련 팀 프로젝트도 같은 팀으로 진행하게 되는데, 다음 프로젝트에서는 체계적으로 코드 버전 관리를 통해 이번에 제대로 하지 못했던 부분들을 채워나갈 수 있도록 노력해보겠다.



- 향후 개선 방향



    > Git 및 Github를 이용하여 코드 버전 관리를 체계적으로 실시

