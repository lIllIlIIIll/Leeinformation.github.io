---
layout: single
permalink: /MachineRunning/
title:  "Logistic Regression Classifier - weatherAUC"
categories: Machine-Running
tag: [python, blog, jekyll, Machine Running, Logistic Regression]
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


> ## 로지스틱 회귀 소개


로지스틱 회귀(Logistic Regression) (또는 Logit Regression)는 데이터 과학의 새로운 이진 분류 문제를 해결할 때, 생각할 수 있는 대표적인 알고리즘이다.



로지스틱 회귀는 이산적인 값을 가지는 클래스를 예측할 때 사용되는 지도 학습 알고리즘으로, 실제로 서로 다른 특성들의 관측치를 분류할 때 사용된다.



여기서 "이산적인 값"이란 불연속적인 값 또는 구분되어 셀 수 있는 것을 의미하는데, 로지스틱 회귀의 결과는 입력 변수에 따라 해당되는 클래스에 대한 롹률을 계산하여 가장 높은 확률을 가진 클래스를 예측하고 이러한 예측은 불연속적인 특성으로 표현된다.


> ## 로지스틱 회귀 분석


통계학에 있어, 로지스틱 회귀 모델은 주로 분류 목적으로 사용되는 통계학 모델에서 폭넓게 사용된다.



관측치의 집합이 주어졌을 때, 로지스틱 회귀 알고리즘은 두 개 이상의 이산적인 클래스로 분류하는데 도움을 주고, 목표 또한 이산적인 형태를 가진다.



다음은 로지스틱 회귀의 작동 방식이다.


> > ### 선형 방정식 구현


로지스틱 회귀는 독립 변수 or 설명 변수로 응답 값을 예측하는 선형 방정식을 구현함으로써 작동한다.



예를 들어, 공부한 시간과 시험 합격률의 예시를 보자.



x1은 공부한 시간(설명 변수)이고, z는 시험 합격률(목표 변수)이다.



만약 하나의 설명 변수(x1) 및 목표 변수(z)를 가지고 있다면, 이에 대한 선형 방정식은 다음과 같다.



```python

z = β0 + β1x1

```



위의 식에서 β0 와 β1은 모델의 파라미터인데, 만약 설명 변수가 여러 개가 주어진다면 다음과 같이 작성할 수 있다.



```python

z = β0 + β1x1+ β2x2+……..+ βnxn

```



여기서 β0, β1, β2 및 βn은 모델의 파라미터이다.



따라서 예측값은 위의 선형 방정식으로 주어지며, z로 표현될 수 있다.



> > ### Sigmoid 함수


예측된 값 z는 0과 1사이의 확률 값으로 변환되는데 이 때 예측 값에서 확률 값으로 변환하기 위해서 sigmoid function을 사용해야 한다.



sigmoid 함수는 주어진 실수 값을 0과 1사이의 확률 값으로 매핑한다.



sigmoid 함수는 S자 모양의 곡선을 가지고 해당 곡선은 sigmoid 곡선이라고도 한다.



다음 그래프는 sigmoid 함수이다.





![image.png](/image/sigmoid function.png)


> > ### 결정 경계


sigmoid 함수는 0과 1사이의 확률 값으로 반환하는데, 해당 확률 값은 0 또는 1의 이산적인 클래스로 매핑된다.



확률 값을 이산적인 클래스로 매핑하기 위해서 임계값을 선택해야 하고, 해당 임계값 보다 높다면 1로 매핑되고, 해당 임계값 보다 낮다면 0으로 매핑된다.



이를 수학적으로 표현하면 다음과 같다.



```python

p ≥ 0.5 → class = 1

p < 0.5 → class = 0

```



일반적으로 임계값은 0.5로 설정된다.



만약 확률 값이 0.8이라면 관측치는 클래스 1로 매핑되고 확률 값이 0.2라면 관측치는 0으로 매핑된다.



이를 그래프로 표현하면 다음과 같다.



![image.png](/image/sigmoid function and Decision Bound.png)


> > ### 예측 함수


로지스틱 회귀에서 sigmoid 함수와 임계값을 이용하여 예측 함수를 만들 수 있다.



예측 함수는 양성(yes 또는 True)관측치의 확률값을 반환하는데 이를 class 1이라 하며 P(class = 1)로 표현한다.



만약 확률이 1로 가까워진다면 이는 관측치가 class 1에 속할 가능성이 높음을 의미하고, 반대의 경우 관측치가 class 0에 속할 가능성이 높음을 의미한다.


> ## 로지스틱 회귀의 가정


로지스틱 회귀 모델은 몇개의 가정을 요구하는데 다음과 같다.



1. 종속 변수가 이항, 다항 또는 순서형일 경우 사용할 수 있다.



2. 관측치가 각각 독립적이어야한다. 이는 반복적인 측정에서 나온 관측치는 사용할 수 없음을 의미한다.



3. 독립 변수 사이에 다중선형성이 적거나 없어야한다. 이는 독립 변수들이 서로 연관성이 적어야 함을 의미한다.



4. 독립 변수와 로그 확률의 선형성을 가정한다.



5. 샘플의 크기가 클수록 높은 정확성을 얻을 수 있다.


> ## 로지스틱 회귀의 유형


로지스틱 회귀 모델은 목표 변수에 따라 세 가지의 유형으로 분류된다.



1. 이항 로지스틱 회귀



이항 로지스틱 회귀에서 목표 변수는 두개의 특성값을 갖는다.



예시로 yes/no, good/bad, true/false, spam/no spam, pass/fail 등이 있다.



2. 다항 로지스틱 회귀



다항 로지스틱 회귀에서 목표 변수는 특정한 순서가 없는 세개 이상의 특성(명목적인 특성)를 갖는다.



예시로 과일 - 사과/망고/오렌지/바나나 가 있다.



3. 순서형 로지스틱 회귀



순서형 로지스틱 회귀에서 목표 변수는 순서가 있는 세개 이상의 특성을 갖는다.



예시로 성적 - 나쁨/보통/좋음/우수 가 있다.


> ## 라이브러리



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

<pre>
/kaggle/input/weatheraus/weatherAUS.csv
</pre>

```python
import warnings

warnings.filterwarnings('ignore')
```

> ## 데이터셋 불러오기



```python
data = '/kaggle/input/weatheraus/weatherAUS.csv'

df = pd.read_csv(data)
```

> ## 데이터셋 분석



```python
# 데이터셋의 차원 확인

df.shape
```

<pre>
(142193, 24)
</pre>
142193개의 인스턴스와 24개의 특성으로 이루어진 데이터셋이다.



```python
# 데이터셋의 첫 5개 정보 확인

df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>...</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RISK_MM</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>...</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>...</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>...</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>0.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>...</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>1.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>...</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>0.2</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>


> > ### RISK_MM 특성 삭제



```python
col_names = df.columns
col_names
```

<pre>
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow'],
      dtype='object')
</pre>

```python
# RISK_MM 특성 삭제

df.drop(["RISK_MM"], axis=1, inplace=True)
```


```python
df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 23 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   Date           142193 non-null  object 
 1   Location       142193 non-null  object 
 2   MinTemp        141556 non-null  float64
 3   MaxTemp        141871 non-null  float64
 4   Rainfall       140787 non-null  float64
 5   Evaporation    81350 non-null   float64
 6   Sunshine       74377 non-null   float64
 7   WindGustDir    132863 non-null  object 
 8   WindGustSpeed  132923 non-null  float64
 9   WindDir9am     132180 non-null  object 
 10  WindDir3pm     138415 non-null  object 
 11  WindSpeed9am   140845 non-null  float64
 12  WindSpeed3pm   139563 non-null  float64
 13  Humidity9am    140419 non-null  float64
 14  Humidity3pm    138583 non-null  float64
 15  Pressure9am    128179 non-null  float64
 16  Pressure3pm    128212 non-null  float64
 17  Cloud9am       88536 non-null   float64
 18  Cloud3pm       85099 non-null   float64
 19  Temp9am        141289 non-null  float64
 20  Temp3pm        139467 non-null  float64
 21  RainToday      140787 non-null  object 
 22  RainTomorrow   142193 non-null  object 
dtypes: float64(16), object(7)
memory usage: 25.0+ MB
</pre>
데이터셋을 불러올 때 설명을 보면 RISK_MM 특성을 삭제하라고 주어진다.


> > ### 특성 확인


데이터셋에 범주형 특성과 수치형 특성이 존재하는데, 이를 분리해야한다.



범주형 특성은 데이터 타입이 object로 주어지고, 수치형 특성은 float64로 주어진다.



```python
# 범주형 특성 탐색

categorical = [var for var in df.columns if df[var].dtype=='O']

print(f'범주형 특성의 개수 : {len(categorical)}개')
print(f'범주형 특성 : {categorical}')
```

<pre>
범주형 특성의 개수 : 7개
범주형 특성 : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>

```python
# 범주형 특성 확인

df[categorical].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>


범주형 특성 요약



* 날짜 데이터가 존재



* 날짜 데이터 이외의 6개의 특성 존재



* 이진 범주형 변수 존재 - RainToday / RainTomorrow



* 목표 변수는 RainTomorrow


> > > #### 범주형 특성 문제 확인



```python
# 결측치가 존재하는지 확인

df[categorical].isnull().sum()
```

<pre>
Date                0
Location            0
WindGustDir      9330
WindDir9am      10013
WindDir3pm       3778
RainToday        1406
RainTomorrow        0
dtype: int64
</pre>
4개의 특성에 결측치가 존재한다는 것을 알 수 있다.



```python
# 빈도수 확인

for var in categorical:
    print(df[var].value_counts())
```

<pre>
2013-12-01    49
2014-01-09    49
2014-01-11    49
2014-01-12    49
2014-01-13    49
              ..
2007-11-29     1
2007-11-28     1
2007-11-27     1
2007-11-26     1
2008-01-31     1
Name: Date, Length: 3436, dtype: int64
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Launceston          3028
Ballarat            3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cairns              2988
Cobar               2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
Penrith             2964
NorfolkIsland       2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
No     109332
Yes     31455
Name: RainToday, dtype: int64
No     110316
Yes     31877
Name: RainTomorrow, dtype: int64
</pre>

```python
# 빈도수의 분포

for var in categorical:
    print(df[var].value_counts()/np.float(len(df)))
```

<pre>
2013-12-01    0.000345
2014-01-09    0.000345
2014-01-11    0.000345
2014-01-12    0.000345
2014-01-13    0.000345
                ...   
2007-11-29    0.000007
2007-11-28    0.000007
2007-11-27    0.000007
2007-11-26    0.000007
2008-01-31    0.000007
Name: Date, Length: 3436, dtype: float64
Canberra            0.024038
Sydney              0.023468
Perth               0.022455
Darwin              0.022448
Hobart              0.022420
Brisbane            0.022230
Adelaide            0.021731
Bendigo             0.021337
Townsville          0.021330
AliceSprings        0.021316
MountGambier        0.021309
Launceston          0.021295
Ballarat            0.021295
Albany              0.021211
Albury              0.021175
PerthAirport        0.021161
MelbourneAirport    0.021161
Mildura             0.021147
SydneyAirport       0.021133
Nuriootpa           0.021112
Sale                0.021098
Watsonia            0.021091
Tuggeranong         0.021084
Portland            0.021070
Woomera             0.021028
Cairns              0.021014
Cobar               0.021014
Wollongong          0.020979
GoldCoast           0.020957
WaggaWagga          0.020929
Penrith             0.020845
NorfolkIsland       0.020845
SalmonGums          0.020782
Newcastle           0.020782
CoffsHarbour        0.020768
Witchcliffe         0.020761
Richmond            0.020753
Dartmoor            0.020697
NorahHead           0.020599
BadgerysCreek       0.020592
MountGinini         0.020444
Moree               0.020071
Walpole             0.019825
PearceRAAF          0.019424
Williamtown         0.017954
Melbourne           0.017125
Nhil                0.011034
Katherine           0.010964
Uluru               0.010697
Name: Location, dtype: float64
W      0.068780
SE     0.065467
E      0.063794
N      0.063526
SSE    0.063245
S      0.062936
WSW    0.062598
SW     0.061867
SSW    0.060552
WNW    0.056726
NW     0.056283
ENE    0.056205
ESE    0.051374
NE     0.049651
NNW    0.046142
NNE    0.045241
Name: WindGustDir, dtype: float64
N      0.080123
SE     0.064434
E      0.063463
SSE    0.063055
NW     0.060144
S      0.059729
W      0.058090
SW     0.057928
NNE    0.055896
NNW    0.055136
ENE    0.054398
ESE    0.053153
NE     0.052935
SSW    0.052380
WNW    0.050593
WSW    0.048125
Name: WindDir9am, dtype: float64
SE     0.074990
W      0.069701
S      0.067500
WSW    0.065608
SW     0.064574
SSE    0.064293
N      0.060952
WNW    0.060875
NW     0.059553
ESE    0.058948
E      0.058667
NE     0.057415
SSW    0.056332
NNW    0.054384
ENE    0.054321
NNE    0.045319
Name: WindDir3pm, dtype: float64
No     0.768899
Yes    0.221213
Name: RainToday, dtype: float64
No     0.775819
Yes    0.224181
Name: RainTomorrow, dtype: float64
</pre>
카디널리티 : 범주형 변수내의 레이블 수



카디널리티가 높다면 머신 러닝에 있어 심각한 문제를 일으킬 수 있는데, 이를 확인한다.



```python
for var in categorical:
    print(var, " contains ", len(df[var].unique()), " labels")
```

<pre>
Date  contains  3436  labels
Location  contains  49  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  2  labels
</pre>
전처리 과정이 필요한 **Date** 특성이 존재한다.



```python
df['Date'].dtypes
```

<pre>
dtype('O')
</pre>
**Date** 특성의 데이터 타입은 object이다.



해당 특성은 현재 날짜로 되어있는데 이를 날짜/시간으로 변경한다.



```python
df['Date'] = pd.to_datetime(df['Date'])
```


```python
# 데이터셋에 연도 추가

df['Year'] = df['Date'].dt.year
df['Year'].head()
```

<pre>
0    2008
1    2008
2    2008
3    2008
4    2008
Name: Year, dtype: int64
</pre>

```python
# 데이터셋에 월 추가

df['Month'] = df['Date'].dt.month
df['Month']
```

<pre>
0         12
1         12
2         12
3         12
4         12
          ..
142188     6
142189     6
142190     6
142191     6
142192     6
Name: Month, Length: 142193, dtype: int64
</pre>

```python
# 데이터셋에 일 추가

df['Day'] = df['Date'].dt.day
df['Day']
```

<pre>
0          1
1          2
2          3
3          4
4          5
          ..
142188    20
142189    21
142190    22
142191    23
142192    24
Name: Day, Length: 142193, dtype: int64
</pre>

```python
# 전체 데이터셋 확인

df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 142193 entries, 0 to 142192
Data columns (total 26 columns):
 #   Column         Non-Null Count   Dtype         
---  ------         --------------   -----         
 0   Date           142193 non-null  datetime64[ns]
 1   Location       142193 non-null  object        
 2   MinTemp        141556 non-null  float64       
 3   MaxTemp        141871 non-null  float64       
 4   Rainfall       140787 non-null  float64       
 5   Evaporation    81350 non-null   float64       
 6   Sunshine       74377 non-null   float64       
 7   WindGustDir    132863 non-null  object        
 8   WindGustSpeed  132923 non-null  float64       
 9   WindDir9am     132180 non-null  object        
 10  WindDir3pm     138415 non-null  object        
 11  WindSpeed9am   140845 non-null  float64       
 12  WindSpeed3pm   139563 non-null  float64       
 13  Humidity9am    140419 non-null  float64       
 14  Humidity3pm    138583 non-null  float64       
 15  Pressure9am    128179 non-null  float64       
 16  Pressure3pm    128212 non-null  float64       
 17  Cloud9am       88536 non-null   float64       
 18  Cloud3pm       85099 non-null   float64       
 19  Temp9am        141289 non-null  float64       
 20  Temp3pm        139467 non-null  float64       
 21  RainToday      140787 non-null  object        
 22  RainTomorrow   142193 non-null  object        
 23  Year           142193 non-null  int64         
 24  Month          142193 non-null  int64         
 25  Day            142193 non-null  int64         
dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
memory usage: 28.2+ MB
</pre>
추가한 특성 **Year**, **Month**, **Day**가 있는 것을 볼 수 있다.



이제 필요없어진 Date 특성을 삭제한다.



```python
df.drop('Date', axis=1, inplace=True)
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


범주형 특성 문제 해결



```python
categorical = [var for var in df.columns if df[var].dtype=='O']

print(f"범주형 특성의 개수 : {len(categorical)}")
print(f"범주형 특성 : {categorical}")
```

<pre>
범주형 특성의 개수 : 6
범주형 특성 : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>
**Location** 특성 문제 해결



```python
# Location 특성 카디널리티 개수 확인

print('Location contains', len(df.Location.unique()), 'labels')
```

<pre>
Location contains 49 labels
</pre>

```python
# 카디널리티 확인

df.Location.unique()
```

<pre>
array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)
</pre>

```python
# 특성 값의 빈도수 확인

df.Location.value_counts()
```

<pre>
Canberra            3418
Sydney              3337
Perth               3193
Darwin              3192
Hobart              3188
Brisbane            3161
Adelaide            3090
Bendigo             3034
Townsville          3033
AliceSprings        3031
MountGambier        3030
Launceston          3028
Ballarat            3028
Albany              3016
Albury              3011
PerthAirport        3009
MelbourneAirport    3009
Mildura             3007
SydneyAirport       3005
Nuriootpa           3002
Sale                3000
Watsonia            2999
Tuggeranong         2998
Portland            2996
Woomera             2990
Cairns              2988
Cobar               2988
Wollongong          2983
GoldCoast           2980
WaggaWagga          2976
Penrith             2964
NorfolkIsland       2964
SalmonGums          2955
Newcastle           2955
CoffsHarbour        2953
Witchcliffe         2952
Richmond            2951
Dartmoor            2943
NorahHead           2929
BadgerysCreek       2928
MountGinini         2907
Moree               2854
Walpole             2819
PearceRAAF          2762
Williamtown         2553
Melbourne           2435
Nhil                1569
Katherine           1559
Uluru               1521
Name: Location, dtype: int64
</pre>

```python
# get_dummies메소드로 원-핫 인코딩

pd.get_dummies(df.Location, drop_first=True).head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>Cobar</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>


**WindGustDir** 특성 문제 해결



```python
# WindGustDir 특성 카디널리티 개수 확인

print('WindGustDir contains', len(df.WindGustDir.unique()), 'labels')
```

<pre>
WindGustDir contains 17 labels
</pre>

```python
# 카디널리티 확인

df['WindGustDir'].unique()
```

<pre>
array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
       'S', 'NW', 'SE', 'ESE', nan, 'E', 'SSW'], dtype=object)
</pre>

```python
# 특성 값의 빈도수 확인

df.WindGustDir.value_counts()
```

<pre>
W      9780
SE     9309
E      9071
N      9033
SSE    8993
S      8949
WSW    8901
SW     8797
SSW    8610
WNW    8066
NW     8003
ENE    7992
ESE    7305
NE     7060
NNW    6561
NNE    6433
Name: WindGustDir, dtype: int64
</pre>

```python
# get_dummies메소드로 원-핫 인코딩
# 결측치가 얼마나 있는지 확인하는 특성 추가

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 행 기준으로 1의 개수 출력

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE    7992
ESE    7305
N      9033
NE     7060
NNE    6433
NNW    6561
NW     8003
S      8949
SE     9309
SSE    8993
SSW    8610
SW     8797
W      9780
WNW    8066
WSW    8901
NaN    9330
dtype: int64
</pre>
**WindGustDir**특성 확인 결과 9330개의 결측치가 존재한다.


**WindDir9am** 특성 문제 해결



```python
# WindDir9am 특성 카디널리티 개수 확인

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

<pre>
WindDir9am contains 17 labels
</pre>

```python
# 카디널리티 확인

df.WindDir9am.unique()
```

<pre>
array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
       'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)
</pre>

```python
# 특성 값의 빈도수 확인

df['WindDir9am'].value_counts()
```

<pre>
N      11393
SE      9162
E       9024
SSE     8966
NW      8552
S       8493
W       8260
SW      8237
NNE     7948
NNW     7840
ENE     7735
ESE     7558
NE      7527
SSW     7448
WNW     7194
WSW     6843
Name: WindDir9am, dtype: int64
</pre>

```python
# 원-핫 인코딩

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 행 기준으로 1의 개수 출력

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     7735
ESE     7558
N      11393
NE      7527
NNE     7948
NNW     7840
NW      8552
S       8493
SE      9162
SSE     8966
SSW     7448
SW      8237
W       8260
WNW     7194
WSW     6843
NaN    10013
dtype: int64
</pre>
**WindDir9am**특성에 10013개의 결측치가 존재한다.


**WindDir3pm** 특성 문제 해결



```python
# WindDir3pm 특성 카디널리티 개수 확인

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

<pre>
WindDir3pm contains 17 labels
</pre>

```python
# 카디널리티 확인

df.WindDir3pm.unique()
```

<pre>
array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
       'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)
</pre>

```python
# 특성 값 빈도수 확인

df['WindDir3pm'].value_counts()
```

<pre>
SE     10663
W       9911
S       9598
WSW     9329
SW      9182
SSE     9142
N       8667
WNW     8656
NW      8468
ESE     8382
E       8342
NE      8164
SSW     8010
NNW     7733
ENE     7724
NNE     6444
Name: WindDir3pm, dtype: int64
</pre>

```python
# 원-핫 인코딩

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 행 기준으로 1의 개수 출력

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     7724
ESE     8382
N       8667
NE      8164
NNE     6444
NNW     7733
NW      8468
S       9598
SE     10663
SSE     9142
SSW     8010
SW      9182
W       9911
WNW     8656
WSW     9329
NaN     3778
dtype: int64
</pre>
**WindDir3pm**특성에는 3778개의 결측치가 존재한다.


**RainToday** 특성 문제 해결



```python
# RainToday 특성 카디널리티 개수 확인

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

<pre>
RainToday contains 3 labels
</pre>

```python
# 카디널리티 확인

df['RainToday'].unique()
```

<pre>
array(['No', 'Yes', nan], dtype=object)
</pre>

```python
# 특성 값 확인

df.RainToday.value_counts()
```

<pre>
No     109332
Yes     31455
Name: RainToday, dtype: int64
</pre>

```python
# 원-핫 인코딩

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 행 기준 1의 개수 출력

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
Yes    31455
NaN     1406
dtype: int64
</pre>
**RainToday**특성에는 1406개의 결측치가 존재한다.


수치형 특성 문제 해결



```python
# 수치형 특성 탐색

numerical = [var for var in df.columns if df[var].dtype!='O']

print(f"수치형 특성의 개수 : {len(numerical)}")
print(f"수치형 특성 : {numerical}")
```

<pre>
수치형 특성의 개수 : 19
수치형 특성 : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
</pre>

```python
df[numerical].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


수치형 특성 요약



* 16개의 수치형 특성 존재(새로 추가한 특성 **Year**, **Month**, **Day** 제외)



* 모든 수치형 변수는 연속적


수치형 특성 문제 해결



```python
# 수치형 특성의 결측치 확인

df[numerical].isnull().sum()
```

<pre>
MinTemp            637
MaxTemp            322
Rainfall          1406
Evaporation      60843
Sunshine         67816
WindGustSpeed     9270
WindSpeed9am      1348
WindSpeed3pm      2630
Humidity9am       1774
Humidity3pm       3610
Pressure9am      14014
Pressure3pm      13981
Cloud9am         53657
Cloud3pm         57094
Temp9am            904
Temp3pm           2726
Year                 0
Month                0
Day                  0
dtype: int64
</pre>
16개의 특성 모두 결측치가 존재한다.



```python
# 이상치 확인

print(round(df[numerical].describe()),2)
```

<pre>
        MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
count  141556.0  141871.0  140787.0      81350.0   74377.0       132923.0   
mean       12.0      23.0       2.0          5.0       8.0           40.0   
std         6.0       7.0       8.0          4.0       4.0           14.0   
min        -8.0      -5.0       0.0          0.0       0.0            6.0   
25%         8.0      18.0       0.0          3.0       5.0           31.0   
50%        12.0      23.0       0.0          5.0       8.0           39.0   
75%        17.0      28.0       1.0          7.0      11.0           48.0   
max        34.0      48.0     371.0        145.0      14.0          135.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
count      140845.0      139563.0     140419.0     138583.0     128179.0   
mean           14.0          19.0         69.0         51.0       1018.0   
std             9.0           9.0         19.0         21.0          7.0   
min             0.0           0.0          0.0          0.0        980.0   
25%             7.0          13.0         57.0         37.0       1013.0   
50%            13.0          19.0         70.0         52.0       1018.0   
75%            19.0          24.0         83.0         66.0       1022.0   
max           130.0          87.0        100.0        100.0       1041.0   

       Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
count     128212.0   88536.0   85099.0  141289.0  139467.0  142193.0   
mean        1015.0       4.0       5.0      17.0      22.0    2013.0   
std            7.0       3.0       3.0       6.0       7.0       3.0   
min          977.0       0.0       0.0      -7.0      -5.0    2007.0   
25%         1010.0       1.0       2.0      12.0      17.0    2011.0   
50%         1015.0       5.0       5.0      17.0      21.0    2013.0   
75%         1020.0       7.0       7.0      22.0      26.0    2015.0   
max         1040.0       9.0       9.0      40.0      47.0    2017.0   

          Month       Day  
count  142193.0  142193.0  
mean        6.0      16.0  
std         3.0       9.0  
min         1.0       1.0  
25%         3.0       8.0  
50%         6.0      16.0  
75%         9.0      23.0  
max        12.0      31.0   2
</pre>
**Rainfall**, **Evaporation**, **WindSpeed9am**, **WindSpeed3pm** 특성에 이상치가 존재한다.



이상치를 boxplot으로 시각화한다.



```python
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

<pre>
Text(0, 0.5, 'WindSpeed3pm')
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABNYAAAMtCAYAAABTh/zrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAADA5klEQVR4nOzde1yUdf7//+dwGoFVFA8cEpUCbROy0taiEK2gMEwis3Na7edja9kqoKWdtI9BWh42Tds2P2qa2n4NqUxNrERcl13F3CQrD6FpQpghKGeG+f3hb+bDCCpMo8Pg4367cWuu6/2aa14jt67bm9f1PhjMZrNZAAAAAAAAAFrEzdkJAAAAAAAAAK6IwhoAAAAAAABgBwprAAAAAAAAgB0orAEAAAAAAAB2oLAGAAAAAAAA2IHCGgAAAAAAAGAHCmsAAAAAAACAHTycnUBrUF9fr6NHj6p9+/YyGAzOTgcAALgAs9mskydPKjg4WG5uPKtsrejnAQCAlmpJP4/CmqSjR48qJCTE2WkAAAAXdPjwYXXv3t3ZaeAs6OcBAAB7NaefR2FNUvv27SWd/gfr0KGDk7MB0JrU1tZq48aNiouLk6enp7PTAdCKlJWVKSQkxNqPQOtEPw/A2dDPA3A2LennUViTrNMCOnToQIcLgI3a2lr5+PioQ4cOdLgANInpha0b/TwAZ0M/D8D5NKefx4IgAAAAAAAAgB0orAEAAAAAAAB2oLAGAAAAAAAA2IHCGgAAAAAAAGAHCmsAAAAAAACAHSisAQAAAAAAAHagsAYAAAAAAADYgcIaAAAAAAAAYAcKawBwFiaTSdnZ2dqyZYuys7NlMpmcnRIAAAAcgH4eAEehsAYATcjIyFBYWJhiY2M1e/ZsxcbGKiwsTBkZGc5ODQAAAL8B/TwAjkRhDQDOkJGRoREjRigyMlI5OTlauXKlcnJyFBkZqREjRtDpAgAAcFH08wA4GoU1AGjAZDIpJSVFCQkJ+vDDD1VVVaXt27erqqpKH374oRISEpSamsp0AQAAABfTsJ+XmZmpgQMHytvbWwMHDlRmZib9PAB2obAGAA3k5OTo4MGDioqKUu/evW2mCPTu3Vs33nijCgoKlJOT4+xUAQAA0AKWft6UKVPk5mb7p7Cbm5smT55MPw9Ai1FYA4AGCgsLJUlTpkxpcorA888/bxMHAAAA12Dpv0VERDTZbjlPPw9AS1BYA4AGunXrJkm66aabmpwicNNNN9nEAQAAwDUEBQVJkvLz85tst5y3xAFAc1BYA4AWMJvNzk4BAAAAdoiOjlavXr2Ulpam+vp6m7b6+nqlp6crNDRU0dHRTsoQgCuisAYADRQXF0uStm7dqsTEROXm5qqyslK5ublKTEzUP/7xD5s4AAAAuAZ3d3fNmjVLa9eubbKft3btWr3xxhtyd3d3dqoAXIiHsxMAgNbEMvQ/PT1df/3rXzVo0CBrW2hoqNLS0jRlyhSmCAAAALigpKQkrV69WikpKY36eatXr1ZSUpITswPgiiisAUADlikC27Zt0969e5Wdna3169crPj5eMTExuueee5giAAAA4MKSkpI0fPhwffnll9Z+3pAhQxipBsAuTAUFgAYaThG45557ZDQadf3118toNOqee+5higAAAEAb4O7urpiYGA0aNEgxMTH07QDYjcIaAJzBMkVg9+7dGjRokB544AENGjRI+fn5TBEAgGbasmWLhg0bpuDgYBkMBmVmZp41dsyYMTIYDJo7d67N+erqao0bN05dunSRr6+v7rrrLh05cuTCJg4AANACFNYAoAlJSUnav3+/srKylJycrKysLO3bt4+iGgA0U3l5ufr166f58+efMy4zM1P/+te/FBwc3Kht/PjxWrNmjVatWqWtW7fq1KlTSkhIkMlkulBpAwAAtAhrrAHAWVimCJSXlzNFAABaKD4+XvHx8eeM+emnn/T000/rs88+05133mnTVlpaqkWLFmnZsmW67bbbJEnLly9XSEiINm3apNtvv/2C5Q4AANBcFNYAAABw0dXX1+uRRx7RxIkT1bdv30bteXl5qq2tVVxcnPVccHCwIiIitG3btrMW1qqrq1VdXW09LisrkyTV1taqtrbWwd8CgCuz3BO4NwA4U0vuCxTWAAAAcNHNmDFDHh4eeuaZZ5psLyoqkpeXlzp16mRzPiAgQEVFRWe9bnp6uqZNm9bo/MaNG+Xj4/PbkgbQJmVlZTk7BQCtTEVFRbNjKawBAADgosrLy9Nf/vIX7dy5UwaDoUXvNZvN53zP5MmTlZycbD0uKytTSEiI4uLi1KFDB7tzBtD21NbWKisrS7GxsfL09HR2OgBaEcuI9+agsAYAAICLKicnR8XFxerRo4f1nMlkUkpKiubOnauDBw8qMDBQNTU1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9O2338pgMOiqq67Stdde6+yUAOA3sayRm5+frxtuuKFRe35+vk0cADQHhTUAaKBbt26SpJtuukkfffSRzdobTz31lGJiYrR161ZrHAC0NcXFxbr//vu1efNmdezYUWazWaWlpRoyZIhWrVqlrl27OjtFALBLdHS0evXqpbS0NGVmZtq01dfXKz09XaGhoYqOjnZOggBcEruCAkALsM4agLZu3LhxKisr0zfffKNff/1VJSUlys/PV1lZmZ555hlnpwcAdnN3d9esWbO0du1aJSYmKjc3V5WVlcrNzVViYqLWrl2rN954g40LALQII9YAoIHi4mJJ0tatWzV8+HDFxsZq3759OnTokLKysvSPf/zDJg4A2poNGzZo06ZN+v3vf289d9VVV+mtt95SXFycEzMDgN8uKSlJq1evVnJysgYNGmQ936tXL61evVpJSUlOzA6AK6KwBgANWNbUsGxesHbtWmtbw80LWHsDQFtVX18vT0/PRuc9PT1VX1/vhIwAwPEMBoOzUwDQRhjMzGtSWVmZ/Pz8VFpaqg4dOjg7HQBOZDKZFBQUpGPHjunOO+/U5Zdfru+//159+vTRDz/8oE8//VTdunXT0aNHmSYAXOLaav9h+PDhOnHihFauXKng4GBJ0k8//aSHHnpInTp10po1a5ycYcu01d8TAPtkZGRoxIgRuvPOO60zE8LDw5WVlaVPP/2UUWsAJLWs/0BhTXS4APwfk8mk4OBgFRcXy2g0qrq62tpmOaawBkBqu/2Hw4cPa/jw4crPz1dISIgMBoN+/PFHRUZG6qOPPlL37t2dnWKLtNXfE4CWM5lMCgsLU5cuXXTs2DEdOnTI2tazZ0917dpVx48f1759++jnAZe4lvQfmAoKAA3k5ORY10+rqamxabMcFxcXKycnR4MHD77Y6QHABRcSEqKdO3cqKytL3333ncxms6666irddtttzk4NAH6TnJwcHTx4UAcPHtSwYcO0fPlyHTlyRN27d9fMmTP1ySefWOPo5wFormYX1q699tpmz0PfuXOn3QkBgDP99NNP1tft2rVTZWVlk8cN4wCgLYqNjVVsbKyz0wAAh7H03+Lj45WZmSmTyaTjx49r4MCByszMVEJCgtavX08/D0CLNLuwlpiYeAHTAIDWoaioyPr61ltv1bPPPmt9kjljxgzrZgYN4wDA1b355pv67//+b7Vr105vvvnmOWOfeeYZh31uXV2dpk6dqvfff19FRUUKCgrS6NGj9cILL8jNzU2SZDabNW3aNL3zzjsqKSnRwIED9dZbb6lv374OywPApeHYsWOSTu8M6ubmJpPJZG1zc3NTYmKi1q9fb40DgOZodmHt5ZdfvpB5AECr8Msvv0iSdYFus9lsfZK5Zs0adevWTSUlJdY4AGgL5syZo4ceekjt2rXTnDlzzhpnMBgcWlibMWOG3n77bS1dulR9+/bVjh079Nhjj8nPz09//vOfJUkzZ87U7NmztWTJEvXu3VvTp09XbGysvv/+e7Vv395huQBo+7p27Srp9AYGjz/+uE1bfX29MjMzbeIAoDlYYw0AGrAM/T9x4oSSkpI0ceJEVVZWKjc3V6+//rpOnDhhEwcAbUFBQUGTry+0f/7znxo+fLjuvPNOSVKvXr20cuVK7dixQ9Lp0Wpz587V888/b92lb+nSpQoICNCKFSs0ZsyYi5YrANd32WWXSZI2bNigxMTERv28DRs22MQBQHM0u7DWqVOnZq+x9uuvv9qdEAA4U0hIiCQpPDxcu3fv1qBBg6xtoaGhCg8P1969e61xANDWvPLKK0pNTZWPj4/N+crKSr3++ut66aWXHPZZN998s95++23t3btXvXv31n/+8x9t3bpVc+fOlXS6yFdUVKS4uDjre4xGo2JiYrRt27YmC2vV1dU2OzqXlZVJkmpra1VbW+uw3AG4nhtuuEG9evWSv7+/vv76a5t+Xq9evXTttdeqpKREN9xwA/cL4BLXkntAswtrlg6OI6WnpysjI0PfffedvL29FRUVpRkzZqhPnz7WmOasq1FdXa3U1FStXLlSlZWVuvXWW7VgwQKX2w4egPPdcsstSktL0969e3XnnXdqwoQJ2rdvn8LDw7Vx40Z9+umn1jgAaIumTZumJ598slFhraKiQtOmTXNoYe3ZZ59VaWmprrzySrm7u8tkMunVV1/VAw88IOn/1rMMCAiweV9AQIAOHTrU5DXT09M1bdq0Ruc3btzY6DsBuPTcd999mjlzpvr376+4uDgZjUZVV1frq6++Ul5eniZNmqTPPvvM2WkCcLKKiopmxza7sDZq1Ci7kjmX7OxsPfXUU7r++utVV1en559/XnFxcdqzZ498fX0lNW9djfHjx+uTTz7RqlWr1LlzZ6WkpCghIUF5eXlyd3d3eN4A2q7Bgwera9euOnbsmL744gtrIU2SvL29JUndunVjC3YAbZbZbG5ylsJ//vMf+fv7O/SzPvjgAy1fvlwrVqxQ3759tWvXLo0fP17BwcE2fc8z8zlbjpI0efJkJScnW4/LysoUEhKiuLg4dejQwaH5A3A9Q4cO1XXXXadJkybpb3/7m/V8r169tGrVKt19991OzA5Aa2EZ8d4cv3mNtcrKykZD5JrbabHMYbdYvHixunXrpry8PA0aNKhZ62qUlpZq0aJFWrZsmW677TZJ0vLlyxUSEqJNmzbp9ttv/61fEcAlxN3dXW+//bbuueeeRm2WP+IWLlxI0R5Am2NZ9sNgMKh37942hSuTyaRTp07pySefdOhnTpw4Uc8995zuv/9+SVJkZKQOHTqk9PR0jRo1SoGBgZJk3THUori4uNEoNguj0Sij0djovKenpzw9PR2aPwDX5OHh0WRx3sPDg/sEAElq0b3ArsJaeXm5nn32Wf3973/X8ePHG7U33La4JUpLSyXJ+jS0Oetq5OXlqba21iYmODhYERER2rZtW5OFNdbeAHAuw4YN0wcffKBJkybZTDXq1q2bZsyYoWHDhnGvANDm7gNz586V2WzW448/rmnTpsnPz8/a5uXlpV69eunGG2906GdWVFTIzc3N5py7u7vq6+slnV7bMjAwUFlZWbr22mslSTU1NcrOztaMGTMcmguAS0NGRoZGjBihhIQELVu2TEeOHFH37t01c+ZMjRgxQqtXr7YO6gCA5rCrsDZp0iR9+eWXWrBggR599FG99dZb+umnn/TXv/5Vr732ml2JmM1mJScn6+abb1ZERISk5q2rUVRUJC8vL3Xq1KlRjOX9Z2LtDQDnYzQaNXv2bO3Zs0clJSXq1KmTrrrqKrm7u2vdunXOTg9AK9CStTdcgWXqZWhoqKKioi7KqI1hw4bp1VdfVY8ePdS3b1999dVXmj17th5//HFJp0cKjx8/XmlpaQoPD1d4eLjS0tLk4+OjBx988ILnB6BtMZlM1mWDMjMzZTKZdPz4cQ0cOFCZmZlKTExUamqqhg8fzuwEAM1mV2Htk08+0XvvvafBgwfr8ccfV3R0tMLCwtSzZ0+9//77euihh1p8zaefflpff/21tm7d2qitJetqNCeGtTcANNcdd9yhrKwsxcbGMjUAgI2WrL3hSmJiYqyvf8uSH80xb948vfjiixo7dqyKi4sVHBysMWPG2GyQMGnSJFVWVmrs2LHWjaw2btxoXWsXAJorJydHBw8e1MqVK+Xm5mYz08rNzU2TJ09WVFSUcnJyWE8XQLPZVVj79ddfFRoaKul05+rXX3+VdHrL9D/96U8tvt64ceP08ccfa8uWLTY7eTZnXY3AwEDV1NRYR5Q0jImKimry81h7A0BzmEwmbdu2TVu2bJGvr6+GDBnC00sAVm21z1BRUaFJkyY5fMmPprRv315z58495+7zBoNBU6dO1dSpUx32uQAuTYWFhZKkiIgImUwmZWdn2/TzLDOnLHEA0Bxu5w9p7PLLL9fBgwclSVdddZX+/ve/Szo9kq1jx47Nvo7ZbNbTTz+tjIwMffHFF9ZinUXDdTUsLOtqWIpm/fv3l6enp01MYWGh8vPzz1pYA4DzycjIUFhYmGJjYzV79mzFxsYqLCxMGRkZzk4NAC6oiRMn6osvvtCCBQtkNBr17rvvatq0aQoODtZ7773n7PQAwG6WwRrz589vsp83f/58mzgAaA6D2Ww2t/RNc+bMkbu7u5555hl9+eWXuvPOO2UymVRXV6fZs2frz3/+c7OuM3bsWK1YsUIfffSR+vTpYz3v5+cnb29vSdKMGTOUnp6uxYsXW9fV2Lx5s77//nvrFIA//elPWrt2rZYsWSJ/f3+lpqbq+PHjysvLa9bokrKyMvn5+am0tJSpoACsi9reeeedio2N1b59+xQeHq6srCx9+umnLGoLQFLb7T/06NHDuuRHhw4dtHPnToWFhWnZsmVauXKly60z2VZ/TwBazmQyKTg4WMXFxUpISNCzzz5r3bxgxowZWrt2rbp166ajR48ySwG4xLWk/2DXVNAJEyZYXw8ZMkTfffedduzYoSuuuEL9+vVr9nUWLlwoSY3mry9evFijR4+W1Lx1NebMmSMPDw+NHDlSlZWVuvXWW7VkyRJuhgBazLKobf/+/bV7926tXbvW2tazZ0/179+fRW0BtGmOXvIDAFqThuNKLK/tGGsCAFbNngrq7++vX375RZL0+OOP6+TJk9a2Hj16KCkpqUVFNen0DaypH0tRTfq/dTUKCwtVVVWl7Oxs69x3i3bt2mnevHk6fvy4Kioq9MknnygkJKRFuQCA9H+L2u7YsUNXX321cnJytHLlSuXk5Ojqq6/Wjh07VFBQoJycHGenCgAXhKOW/ACA1iYnJ0fHjh1Tenq68vPzNWjQID3wwAMaNGiQvvnmG6Wlpam4uJh+HoAWaXZhraamxrr71dKlS1VVVXXBkgIAZ/npp58kSfHx8crMzNTAgQPl7e1t3YY9Pj7eJg4A2prHHntM//nPfySd3kndstbahAkTNHHiRCdnBwD2s2xK8PTTT2v//v3KyspScnKysrKytG/fPj399NM2cQDQHM2eCnrjjTcqMTFR/fv3l9ls1jPPPGNdB+1M//u//+uwBAHgYjp27JgkKSkpSWazudFuUYmJiVq/fr01DgDaGkct+QEArY1lU4L8/Hxdf/31jdrz8/Nt4gCgOZpdWFu+fLnmzJmjAwcOyGAwqLS0lFFrANqcrl27SpIWLFig6dOn69ChQ5Kk2bNnq2fPnvL397eJA4C2pLa2VnFxcfrrX/+q3r17Szq95EePHj2cnBkA/HbR0dHq1auXxo0bp19++cU67X327Nnq1auXunTpotDQUEVHRzs3UQAupdmFtYCAAL322muSpNDQUC1btkydO3e+YIkBgDNcdtllkqSvvvqqUduhQ4eshTZLHAC0JZ6ensrPz5fBYHB2KgDgcO7u7rr33nv1+uuvKyAgQAsXLlS7du1UVVWlqVOnaseOHZo4cSIbVAFoEYOZLVDYhh2AVU1Njdq1a3fO3aEMBoOqqqrk5eV1ETMD0Nq01f5DSkqKPD09rQ9UXV1b/T0BaDmTyaSwsDB16dJFx44dsz4wlWQdsXb8+HHt27eP4hpwiWtJ/6HZI9bO9Pnnn+vzzz9XcXGx6uvrbdpYYw2Aq8rOzrYW1by8vJSUlCRvb29VVlYqIyNDNTU11rXXYmNjnZwtADheTU2N3n33XWVlZWnAgAHy9fW1aZ89e7aTMgOA38ay+/vKlSt13XXXad68efriiy90yy23aNy4ccrLy1NUVJRycnI0ePBgZ6cLwEXYVVibNm2aXnnlFQ0YMEBBQUFMFwDQZixevFiS1K5dO3Xr1k2rVq2ytvXs2VM///yzqqqqtHjxYgprANqk/Px8XXfddZKkvXv32rTR5wPgyiy7fR44cEAPPPCAdY21devWaf78+Zo+fbpNHAA0h12FtbfffltLlizRI4884uh8AMCpdu/eLUn6r//6L82YMUPJycnKzc3VDTfcoNmzZys1NVULFiywxgFAW/Pll186OwUAuCAsu30+8sgjateunU3bzz//bP37ll1BAbSEXYW1mpoaRUVFOToXAHA6y/z5FStWaMGCBTKZTJKkXbt26W9/+5v8/Pxs4gCgLTty5IgMBgMbtgBoE6KiouTm5qb6+noNHjxYYWFh+v7779WnTx/t379f69evl5ubG3/rAmgRN3ve9Mc//lErVqxwdC4A4HSJiYmSpOPHj8vd3V333XefRo8erfvuu0/u7u769ddfbeIAoK2pr6/XK6+8Ij8/P/Xs2VM9evRQx44d9T//8z+N1tUFAFeSk5NjvY9t2LBB8+bN08aNGzVv3jxt2LBB0ul7YE5OjjPTBOBi7BqxVlVVpXfeeUebNm3S1VdfLU9PT5t2FrUF4Kr+9Kc/adKkSZJOj8794IMPzhoHAG3R888/r0WLFum1117TTTfdJLPZrH/84x+aOnWqqqqq9Oqrrzo7RQCwy+bNm62vDQaDzS7wbm5u1pkKmzdv1q233nqx0wPgouwqrH399de65pprJJ1e4LYhFrUF4MrefffdZseNHz/+wiYDAE6wdOlSvfvuu7rrrrus5/r166fLLrtMY8eOpbAGwGVZCmedOnVSYWGhcnJytH79esXHxys6OlpBQUEqKSmxxgFAc9hVWGNRWwBt1b59+6yvz3yS2fC4YRwAtCW//vqrrrzyykbnr7zySut0eABwRSdOnJAkde7cWZ6enoqJiVF5ebliYmLk7u4uf39/lZSUWOMAoDnsWmMNANoqS+EsMDCw0WLdl112mQIDA23iAKCt6devn+bPn9/o/Pz589WvXz8nZAQAjmGZXbV//34NHz5cubm5qqysVG5uroYPH64DBw7YxAFAczR7xFpSUpKWLFmiDh06KCkp6ZyxGRkZvzkxAHCGjh07SpKKiork7e1t03b8+HFVVlbaxAFAWzNz5kzdeeed2rRpk2688UYZDAZt27ZNhw8f1rp165ydHgDYLTw83Pr6888/19q1a63HPj4+TcYBwPk0e8San5+ftXLv5+d3zh8AcFUNn1CaTCbdd999euyxx3TffffZrLfBk0wAbVVMTIz27t2ru+++WydOnNCvv/6qpKQkff/994qOjnZ2egBgt7Fjx8rDw0N+fn7q0qWLTVuXLl3k5+cnDw8PjR071kkZAnBFzR6xtnjx4iZfA0BbYhmJ5u7u3uSuoO7u7jKZTIxYA9CmBQcHs0kBgDbHy8tLEyZM0Ouvv66TJ0/atB05ckT19fWaOHGivLy8nJQhAFdk1+YFANBWWRarPdtuUJbzLGoLoC0rKSnRokWL9O2338pgMOj3v/+9HnvsMfn7+zs7NQD4TW644QZJUn19vc15y7GlHQCay+7NC1avXq2RI0fqhhtu0HXXXWfzAwAAANeUnZ2t0NBQvfnmmyopKdGvv/6qN998U6GhocrOznZ2egBgN5PJpCeffFKSGo1Ksxz/6U9/OusDVgBoil2FtTfffFOPPfaYunXrpq+++kp/+MMf1LlzZ/3www+Kj493dI4AcNE0d51I1pME0FY99dRTGjlypAoKCpSRkaGMjAz98MMPuv/++/XUU085Oz0AsNvmzZt17NgxSY3Xy7UcFxcXa/PmzRc7NQAuzK7C2oIFC/TOO+9o/vz58vLy0qRJk5SVlaVnnnlGpaWljs4RAC6ar7/+2qFxAOBqDhw4oJSUFLm7u1vPubu7Kzk5WQcOHHBiZgDw23zxxRfW1zU1NTZtDY8bxgHA+di1xtqPP/6oqKgoSZK3t7d14cdHHnlEN9xwg+bPn++4DAHgIjpzIdvfGgcArua6667Tt99+qz59+tic//bbb3XNNdc4JykAcIBDhw5ZX3ft2lUPPfSQysvL5evrq/fff1/FxcWN4gDgfOwqrAUGBur48ePq2bOnevbsqdzcXPXr108FBQUym82OzhEALppz3cMMBoO1nXsdgLbqmWee0Z///Gft37/fuoh3bm6u3nrrLb322ms2I3avvvpqZ6UJAC1WV1cnSXJzc1O7du00Z84ca1uPHj3k5uam+vp6axwANIddhbVbbrlFn3zyia677jo98cQTmjBhglavXq0dO3YoKSnJ0TkCQKtAMQ3ApeCBBx6QJE2aNKnJNstDBoPBwALfAFzK8ePHJZ3eAbSqqkoTJkxQRUWFfHx89P7771t3BrXEAUBz2FVYe+edd6w3nSeffFL+/v7aunWrhg0bprvvvtuhCQLAxVRZWenQOABwNQUFBc5OAQAuCB8fH+vrY8eO2YxYa7iZQcM4ADgfuwprbm5ucnP7v30PRo4cqUGDBunVV19V7969+YMTgMvq0qWLQ+MAwNX07NnT2SkAwAURHBxsfd1wiY8zjxvGAcD5tGhX0BMnTuihhx5S165dFRwcrDfffFP19fV66aWXdMUVVyg3N1f/+7//e6FyBYALruEueI6IAwBXtWfPHm3YsEEff/yxzQ8AuCrLupGS5OFhO8ak4XHDOAA4nxaNWJsyZYq2bNmiUaNGacOGDZowYYI2bNigqqoqrVu3TjExMRcqTwC4KJr7hJInmQDaqh9++EF33323du/ebTOCwzJNinXVALiqkpIS6+uamhqbtobHDeMA4HxaNGLt008/1eLFi/XGG2/o448/ltlsVu/evfXFF19QVAPQJnz33XcOjQMAV/PnP/9ZoaGh+vnnn+Xj46NvvvlGW7Zs0YABA7R582ZnpwcAduvatatD4wBAamFh7ejRo7rqqqskSZdffrnatWunP/7xjxckMQBwhp9++sn6uuEitmceN4wDgLbkn//8p1555RV17drVuq7uzTffrPT0dD3zzDMO/7yffvpJDz/8sDp37iwfHx9dc801ysvLs7abzWZNnTpVwcHB8vb21uDBg/XNN984PA8AbV9gYKBD4wBAamFhrb6+Xp6entZjd3d3+fr6OjwpAHCWkydPnrWtYWHtXHEA4MpMJpN+97vfSTq9UcvRo0clnd7U4Pvvv3foZ5WUlOimm26Sp6en1q9frz179mjWrFnq2LGjNWbmzJmaPXu25s+fr+3btyswMFCxsbHchwG0WHOnsjPlHUBLtGiNNbPZrNGjR8toNEqSqqqq9OSTTzYqrmVkZDguQwC4iPr06aN9+/ZJOr2OWsORacHBwTpy5Ig1DgDaooiICH399de6/PLLNXDgQM2cOVNeXl565513dPnllzv0s2bMmKGQkBAtXrzYeq5Xr17W12azWXPnztXzzz+vpKQkSdLSpUsVEBCgFStWaMyYMQ7NB0Dblp2d3ey4uLi4C5wNgLaiRYW1UaNG2Rw//PDDv+nDt2zZotdff115eXkqLCzUmjVrlJiYaG0fPXq0li5davOegQMHKjc313pcXV2t1NRUrVy5UpWVlbr11lu1YMECde/e/TflBuDS1PDeceZ0T0tR7cw4AGhLXnjhBZWXl0uSpk+froSEBEVHR6tz58764IMPHPpZH3/8sW6//Xbde++9ys7O1mWXXaaxY8fqv/7rvyRJBQUFKioqsvkD12g0KiYmRtu2bWuysFZdXa3q6mrrcVlZmSSptrZWtbW1Ds0fgGuxPDxtThz3C+DS1pJ7QIsKaw2fJjpCeXm5+vXrp8cee0z33HNPkzF33HGHzed6eXnZtI8fP16ffPKJVq1apc6dOyslJUUJCQnKy8uTu7u7Q/MF0PZFRUXp7bffblYcALRFt99+u/X15Zdfrj179ujXX39Vp06dGq09+Vv98MMPWrhwoZKTkzVlyhT9+9//1jPPPCOj0ahHH31URUVFkqSAgACb9wUEBOjQoUNNXjM9PV3Tpk1rdH7jxo3y8fFxaP4AXMuOHTusr728vGx2Am14vGPHDq1bt+6i5weg9aioqGh2bIsKa44WHx+v+Pj4c8YYjcazLh5ZWlqqRYsWadmyZbrtttskScuXL1dISIg2bdpk0zEEgOYIDg52aBwAuJqlS5dqxIgRNkt9+Pv7X5DPqq+v14ABA5SWliZJuvbaa/XNN99o4cKFevTRR61xZxb0zGbzWYt8kydPVnJysvW4rKxMISEhiouLU4cOHS7AtwDgKlJTU62vzxyN0vDY3d1dQ4cOvWh5AWh9LCPem8OphbXm2Lx5s7p166aOHTsqJiZGr776qrp16yZJysvLU21trc30gODgYEVERGjbtm1nLawxRQDA2TR8cnm+OO4XwKWtrd4DUlNTNXbsWA0bNkwPP/yw7rjjDnl4XJguY1BQkHXHeYvf//73+vDDDyX93858RUVFCgoKssYUFxc3GsVmYTQaresBN+Tp6WmzCReAS0/79u2tr81ms01bw+P27dtzvwAucS25B7Tqwlp8fLzuvfde9ezZUwUFBXrxxRd1yy23KC8vT0ajUUVFRfLy8lKnTp1s3hcQEGCdOtAUpggAOJv33nuvWXHvvvtus4twANqmlkwRcCWFhYXasGGDVq5cqfvvv1/e3t6699579fDDDzt8GvxNN93UaKfRvXv3qmfPnpKk0NBQBQYGKisrS9dee62k0w82srOzNWPGDIfmAqDt6927t7766qtmxQFAc7Xqwtp9991nfR0REaEBAwaoZ8+e+vTTT607QzXlXNMDJKYIADi7N998s1lxJ06cYIoAcIlryRQBV+Lh4aGEhAQlJCSooqJCa9as0YoVKzRkyBB1795dBw4ccNhnTZgwQVFRUUpLS9PIkSP173//W++8847eeecdSaengI4fP15paWkKDw9XeHi40tLS5OPjowcffNBheQC4NDR3DW7W6gbQEq26sHamoKAg9ezZ07qbS2BgoGpqalRSUmIzaq24uPic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nztnzhwVFhZq165d2rt3bwuzBgAAuDQ07Ks1/LnjjjucnZpDnPng1GL79u367//+74ufEICLzsPZCQDAb3XHHXdo8eLFqqur0549e/T444/rxIkTWrlyZbPe7+fn1+LPPHDggPr376/w8PAWvxcAAOBSYumrNWQ0Gp2UTfPU1NTIy8vL7vd37drVgdkAaM0YsQbA5RmNRgUGBqp79+6Ki4vTfffdp40bN0qSTCaTnnjiCYWGhsrb21t9+vTRX/7yF5v3nzkVdPDgwXrmmWc0adIk+fv7KzAwUFOnTrW29+rVSx9++KHee+89GQwGjR49WpI0e/ZsRUZGytfXVyEhIRo7dqxOnTp1ob8+AABAq2bpqzX86dSpkx544AHdf//9NrG1tbXq0qWLtRC3YcMG3XzzzerYsaM6d+6shIQEHThwwBp/8OBBGQwGrVq1SlFRUWrXrp369u2rzZs321w3Oztbf/jDH2Q0GhUUFKTnnntOdXV11vbBgwfr6aefVnJysrp06aLY2FhJ5+7fbd68WY899phKS0utI/EsfcYzp4L++OOPGj58uH73u9+pQ4cOGjlypH7++Wdru2WJkGXLlqlXr17y8/PT/fffr5MnT/7mf38AFxaFNQBtyg8//KANGzbI09NTklRfX6/u3bvr73//u/bs2aOXXnpJU6ZM0d///vdzXmfp0qXy9fXVv/71L82cOVOvvPKKsrKyJJ0e2n/HHXdo5MiRKiwstBbq3Nzc9Oabbyo/P19Lly7VF198oUmTJl3YLwwAAOCiHnroIX388cc2DyI/++wzlZeX65577pEklZeXKzk5Wdu3b9fnn38uNzc33X333aqvr7e51sSJE5WSkqKvvvpKUVFRuuuuu3T8+HFJ0k8//aShQ4fq+uuv13/+8x8tXLhQixYt0vTp022usXTpUnl4eOgf//iH/vrXv0o6d/8uKipKc+fOVYcOHVRYWKjCwkKlpqY2+p5ms1mJiYn69ddflZ2draysLB04cED33XefTdyBAweUmZmptWvXau3atcrOztZrr732G/+VAVxoTAUF4PLWrl2r3/3udzKZTKqqqpJ0+umiJHl6emratGnW2NDQUG3btk1///vfNXLkyLNe8+qrr9bLL78sSQoPD9f8+fP1+eefKzY2Vl27dpXRaJS3t7cCAwOt7xk/frzN5/zP//yP/vSnP2nB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vHn64osvtH//fo0bN05eXl7OTguAC2LEGgAAAADgkjFp0iT5+voqNTVV69atU2pqqnx9fTVp0iRnpwbABTFiDQAAAABwSZg0aZJef/11BQQEaNq0aTIajaqurtbLL7+s119/XZI0c+ZMJ2cJwJUwYg0AAAAA0ObV1NRozpw5CggI0JEjR/T444+rU6dOevzxx3XkyBEFBARozpw5qqmpcXaqAFwIhTUAAAAAQJu3YMEC1dXVafr06fLwsJ285eHhoVdeeUV1dXVasGCBkzIE4IoorAEAAAAA2rwDBw5IkhISEppst5y3xAFAc1BYAwAAAAC0eVdccYUkae3atU22W85b4gCgOSisAQAAAADavLFjx8rDw0MvvPCC6urqbNrq6ur00ksvycPDQ2PHjnVShgBcEYU1AAAAAECb5+XlpQkTJujnn39W9+7d9e677+rXX3/Vu+++q+7du+vnn3/WhAkT5OXl5exUAbgQj/OHAAAAAADg+mbOnClJmjNnjs3INA8PD02cONHaDgDNxYg1AAAAAMAlY+bMmSovL9cbb7yhoUOH6o033lB5eTlFNQB2cWphbcuWLRo2bJiCg4NlMBiUmZlpbautrdWzzz6ryMhI+fr6Kjg4WI8++qiOHj1qc43q6mqNGzdOXbp0ka+vr+666y4dOXLkIn8TAAAAAICr8PLy0jPPPKP//u//1jPPPMP0TwB2c2phrby8XP369dP8+fMbtVVUVGjnzp168cUXtXPnTmVkZGjv3r266667bOLGjx+vNWvWaNWqVdq6datOnTqlhIQEmUymi/U1AAAAYIe6ujq98MILCg0Nlbe3ty6//HK98sorqq+vt8aYzWZNnTpVwcHB8vb21uDBg/XNN984MWsAAID/49Q11uLj4xUfH99km5+fn7KysmzOzZs3T3/4wx/0448/qkePHiotLdWiRYu0bNky3XbbbZKk5cuXKyQkRJs2bdLtt99+wb8DAAAA7DNjxgy9/fbbWrp0qfr27asdO3bosccek5+fn/785z9LOj1la/bs2VqyZIl69+6t6dOnKzY2Vt9//73at2/v5G8AAAAudS61eUFpaakMBoM6duwoScrLy1Ntba3i4uKsMcHBwYqIiNC2bdvOWlirrq5WdXW19bisrEzS6emntbW1F+4LAHA5lnsC9wYAZ+K+8Nv985//1PDhw3XnnXdKknr16qWVK1dqx44dkk6PVps7d66ef/55JSUlSZKWLl2qgIAArVixQmPGjHFa7gAAAJILFdaqqqr03HPP6cEHH1SHDh0kSUVFRfLy8lKnTp1sYgMCAlRUVHTWa6Wnp2vatGmNzm/cuFE+Pj6OTRxAm3DmCFoAqKiocHYKLu/mm2/W22+/rb1796p37976z3/+o61bt2ru3LmSpIKCAhUVFdk8RDUajYqJidG2bduaLKzxABVAc/EAFcDZtOS+4BKFtdraWt1///2qr6/XggULzhtvNptlMBjO2j558mQlJydbj8vKyhQSEqK4uDhr0Q4ApNP3n6ysLMXGxsrT09PZ6QBoRSwFG9jv2WefVWlpqa688kq5u7vLZDLp1Vdf1QMPPCBJ1gelAQEBNu8LCAjQoUOHmrwmD1ABtBQPUAGcqSUPUFt9Ya22tlYjR45UQUGBvvjiC5vCV2BgoGpqalRSUmIzaq24uFhRUVFnvabRaJTRaGx03tPTkz+cATSJ+wOAM3FP+O0++OADLV++XCtWrFDfvn21a9cujR8/XsHBwRo1apQ17swHpud6iMoDVADNxQNUAGfTkgeorbqwZimq7du3T19++aU6d+5s096/f395enoqKytLI0eOlCQVFhYqPz9fM2fOdEbKAAAAaKaJEyfqueee0/333y9JioyM1KFDh5Senq5Ro0YpMDBQ0umRa0FBQdb3FRcXNxrFZsEDVADNYTKZtG3bNm3ZskW+vr4aMmSI3N3dnZ0WgFaiJX0GtwuYx3mdOnVKu3bt0q5duySdXkdj165d+vHHH1VXV6cRI0Zox44dev/992UymVRUVKSioiLV1NRIOr1z6BNPPKGUlBR9/vnn+uqrr/Twww8rMjLSuksoAAAAWqeKigq5udl2R93d3VVfXy9JCg0NVWBgoM00rZqaGmVnZ59zdgIAnEtGRobCwsIUGxur2bNnKzY2VmFhYcrIyHB2agBckFNHrO3YsUNDhgyxHluG7Y8aNUpTp07Vxx9/LEm65pprbN735ZdfavDgwZKkOXPmyMPDQyNHjlRlZaVuvfVWLVmyhKcNAAAArdywYcP06quvqkePHurbt6+++uorzZ49W48//rik01NAx48fr7S0NIWHhys8PFxpaWny8fHRgw8+6OTsAbiijIwMjRgxQgkJCVq2bJmOHDmi7t27a+bMmRoxYoRWr15t3YUYAJrDYDabzc5OwtnKysrk5+en0tJS1t4AYKO2tlbr1q3T0KFDmUIEwAb9h9/u5MmTevHFF7VmzRoVFxcrODhYDzzwgF566SV5eXlJOr2e2rRp0/TXv/5VJSUlGjhwoN566y1FREQ06zP4PQGwMJlMCgsLU2RkpDIzM2Uymaz9PHd3dyUmJio/P1/79u1joAZwiWtJ/6FVr7EGAACAtqt9+/aaO3eu5s6de9YYg8GgqVOnaurUqRctLwBtU05Ojg4ePKiVK1fKzc1NJpPJ2ubm5qbJkycrKipKOTk51hlSAHA+Tl1jDQAAAACAi6GwsFCSzjri1XLeEgcAzUFhDQAAAADQ5ll2F87Pz2+y3XK+4S7EAHA+FNYAAAAAAG1edHS0evXqpbS0NFVVVenNN9/UO++8ozfffFNVVVVKT09XaGiooqOjnZ0qABfCGmsAAAAAgDbP3d1ds2bN0j333CMfHx9Z9vFbt26dJk6cKLPZrA8//JCNCwC0CCPWAAAAAACXhNzcXEmnN0ZpyM3NzaYdAJqLwhoAAAAAoM2rqanRnDlzFBAQoIqKCmVlZSk5OVlZWVkqLy9XQECA5syZo5qaGmenCsCFUFgDAAAAALR5CxYsUF1dnaZPny6j0aiYmBgNGjRIMTExMhqNeuWVV1RXV6cFCxY4O1UALoTCGgAAAACgzTtw4IAkKSEhocl2y3lLHAA0B4U1AAAAAECbd8UVV0iS1q5d22S75bwlDgCag8IaAAAAAKDNGzt2rDw8PPTCCy+orq7Opq2urk4vvfSSPDw8NHbsWCdlCMAVUVgDAAAAALR5Xl5emjBhgn7++Wd1795dkyZN0rp16zRp0iR1795dP//8syZMmCAvLy9npwrAhXg4OwEAAAAAAC6GmTNnau/evfroo480d+5cm7bhw4dr5syZzkkMgMuisAYAAAAAuCRkZGTo448/1p133qnLL79c33//vfr06aMffvhBH3/8sTIyMpSUlOTsNAG4EAprAAAAAIA2z2QyKSUlRQkJCcrMzJTJZNK6des0dOhQubu7KzExUampqRo+fLjc3d2dnS4AF8EaawAAAACANi8nJ0cHDx7UlClT5OZm+6ewm5ubJk+erIKCAuXk5DgpQwCuiMIaAAAAAKDNKywslCRFREQ02W45b4kDgOagsAYAAAAAaPOCgoIkSfn5+U22W85b4gCgOSisAQAAAADavOjoaPXq1UtpaWmqqqrSm2++qXfeeUdvvvmmqqqqlJ6ertDQUEVHRzs7VQAuhM0LAAAAAABtnru7u2bNmqV77rlHPj4+MpvNkqR169Zp4sSJMpvN+vDDD9m4AECLMGINAAAAAHBJyM3NlSQZDAab85bNDCztANBcFNYAAAAAAG1eTU2N5syZo4CAAFVUVCgrK0vJycnKyspSeXm5AgICNGfOHNXU1Dg7VQAuhMIaAAAAAKDNW7Bggerq6jR9+nQZjUbFxMRo0KBBiomJkdFo1CuvvKK6ujotWLDA2akCcCEU1gAAAAAAbd6BAwckSQkJCU22W85b4gCgOSisAQAAAADavCuuuEKStHbt2ibbLectcQDQHBTWAAAAAABt3tixY+Xh4aEXXnhBdXV1Nm11dXV66aWX5OHhobFjxzopQwCuiMIaAAAAAKDN8/Ly0oQJE/Tzzz+re/fuevfdd/Xrr7/q3XffVffu3fXzzz9rwoQJ8vLycnaqAFyIh7MTAAAAAADgYpg5c6Ykac6cOTYj0zw8PDRx4kRrOwA0FyPWAAAAAACXjJkzZ6q8vFxvvPGGhg4dqjfeeEPl5eUU1QDYhcIaAAAAAOCSYjKZtH//fh09elT79++XyWRydkoAXBSFNQAAAADAJSMxMVE+Pj56++23tWvXLr399tvy8fFRYmKis1MD4IIorAEAAAAALgmJiYn66KOP5OXlpUmTJmnhwoWaNGmSvLy89NFHH1FcA9BiFNYAAAAAAG1eZWWltah28uRJTZ8+XUFBQZo+fbpOnjxpLa5VVlY6O1UALoTCGgAAAACgzZs4caIkKTk5WV5eXjZtXl5eGj9+vE0cADQHhTUAAAAAQJu3b98+SdIf//jHJtufeOIJmzgAaA4KawAAAACANi88PFyS9O677zbZvmjRIps4AGgOCmsAAAAAgDbv9ddflyTNnj1blZWVys7O1pYtW5Sdna3KykrNnTvXJg4AmsPD2QkAAAAAAHCheXt7a/jw4froo4/k4+NjPT979mzr6+HDh8vb29sZ6QFwUU4dsbZlyxYNGzZMwcHBMhgMyszMtGk3m82aOnWqgoOD5e3trcGDB+ubb76xiamurta4cePUpUsX+fr66q677tKRI0cu4rcAAAAAALiCRx999De1A8CZnFpYKy8vV79+/TR//vwm22fOnKnZs2dr/vz52r59uwIDAxUbG6uTJ09aY8aPH681a9Zo1apV2rp1q06dOqWEhASZTKaL9TUAAAAAAK2cyWRSSkqKhg0bplOnTunJJ5/UNddcoyeffFKnTp3SsGHDlJqayt+SAFrEqVNB4+PjFR8f32Sb2WzW3Llz9fzzzyspKUmStHTpUgUEBGjFihUaM2aMSktLtWjRIi1btky33XabJGn58uUKCQnRpk2bdPvtt1+07wIAAAAAaL1ycnJ08OBBrVy5Ur6+vnrzzTe1bt06DR06VJ6enpo8ebKioqKUk5OjwYMHOztdAC6i1a6xVlBQoKKiIsXFxVnPGY1GxcTEaNu2bRozZozy8vJUW1trExMcHKyIiAht27btrIW16upqVVdXW4/LysokSbW1taqtrb1A3wiAK7LcE7g3ADgT9wUAcC2FhYWSpIiIiCbbLectcQDQHK22sFZUVCRJCggIsDkfEBCgQ4cOWWO8vLzUqVOnRjGW9zclPT1d06ZNa3R+48aNNotYAoBFVlaWs1MA0MpUVFQ4OwUAQAsEBQVJkvLz83XDDTc0as/Pz7eJA4DmaLWFNQuDwWBzbDabG5070/liJk+erOTkZOtxWVmZQkJCFBcXpw4dOvy2hAG0KbW1tcrKylJsbKw8PT2dnQ6AVsQy4h0A4Bqio6PVq1cvpaWlNdo4r76+Xunp6QoNDVV0dLRzEgTgklptYS0wMFDS6VFpDZ8YFBcXW0exBQYGqqamRiUlJTaj1oqLixUVFXXWaxuNRhmNxkbnPT09+cMZQJO4PwA4E/cEAHAt7u7umjVrlkaMGKHBgwcrJyfH2hYdHa2tW7dq9erVcnd3d2KWAFyNU3cFPZfQ0FAFBgbaTL+qqalRdna2tWjWv39/eXp62sQUFhYqPz//nIU1AAAAAMClJykpSWaz2aaoJp3e2MBsNls3zgOA5nLqiLVTp05p//791uOCggLt2rVL/v7+6tGjh8aPH6+0tDSFh4crPDxcaWlp8vHx0YMPPihJ8vPz0xNPPKGUlBR17txZ/v7+Sk1NVWRkpHWXUAAAAAAApMZLDV199dX6+uuvbdrNZvPFTguAC3NqYW3Hjh0aMmSI9diy7tmoUaO0ZMkSTZo0SZWVlRo7dqxKSko0cOBAbdy4Ue3bt7e+Z86cOfLw8NDIkSNVWVmpW2+9VUuWLGH4LgAAAADAKi8vz/r6wIEDCgkJ0bp16zR06FAdPnxYV1xxhTWuf//+zkoTgIsxmCnHq6ysTH5+fiotLWXzAgA2amtrrR0u1lMC0BD9B9fA7wmAhWW0msFgUH19faN+npubm3W0Gn8mA5e2lvQfWu0aawAAAAAAONrYsWObPP/4449f5EwAtAUU1gAAAAAAl4wFCxY0ef5///d/L3ImANoCCmsAAAAAgDZvx44dkk5P81y/fr28vb2VmJgob29vrV+/3jr90xIHAM3h1M0LgP+vvXuPi6re9z/+HgEHEMRbAioqoYmmpqlZXgLqiGl1NO22Tctsty3NS5SUmYmlkFpqFy9Hz9ZtndB9yrJ2GxN/bUHMTMQ0NbU08m7W1kS5X9bvDw8TE6jMNLAGeD0fDx/O+q4vyzfwcD0+85m11hcAAAAAqkPZBQkGDx5se11cXGy3zcIFABzBFWsAAAAAAACAE2isAQAAAABqvYMHD9pef/LJJ3b7ym6XnQcAV0NjDQAAAABQ611//fWSJKvVqrvvvlsFBQVat26dCgoKdPfdd8tqtdrNA4DKoLEGAAAAAKj1iouLJUnTp0+vcH9sbKzdPACoDBprAAAAAIBaz8PDQ5L0yiuvVLh/7ty5dvMAoDJorAEAAAAAar19+/ZJkvLz87Vu3TrVr19fQ4cOVf369bVu3Trl5+fbzQOAyqCxBgAAAACo9Tp06GB7fc8999jtK7tddh4AXA2NNQAAAAAAAMAJNNYAAAAAALXe9u3bba8//fRT27PUPDw89Omnn1Y4DwCuxtPsAAAAAAAAVLXevXvbXt95553Kzc1VUlKSBg8eLC8vL7t5hmGYERFADcQVawAAADDNiRMnNHLkSDVt2lS+vr7q1q2bMjIybPsNw1BcXJxatGghHx8fRUZG8mBxAH/I448/XuH4qFGjqjkJgNqAxhoAAABMce7cOfXt21deXl5av369vv32W73++utq1KiRbc7cuXM1f/58vf3220pPT1dQUJAGDBigCxcumBccQI22fPnyCsfffffdak4CoDagsQYAAABTzJkzRyEhIVq5cqVuuukmtW3bVrfffrvCwsIkXbpabeHChZo2bZqGDRumzp07a9WqVcrJyVFiYqLJ6QHUNF999ZXt9aFDh+z2ld0uOw8AroZnrAEAAMAUn3zyiQYOHKj77rtPqampatmypcaNG2e7TSszM1OnT59WdHS07WusVqsiIiK0detWjR07ttwx8/PzlZ+fb9vOysqSJBUWFqqwsLCKvyMA7qx79+621+3bt7/iPM4XQN3myDmAxhoAAABM8cMPP2jJkiWKiYnRCy+8oO3bt2vixImyWq16+OGHdfr0aUlSYGCg3dcFBgbqyJEjFR4zISFBM2fOLDeenJwsX19f138TAGqUdevWaejQoVfcn5SUVH2BALilnJycSs+lsQYAl1FQUKC33npL//rXv3To0CFNmDBB9evXNzsWANQaJSUl6tmzp+Lj4yVdukpk3759WrJkiR5++GHbPIvFYvd1hmGUGys1depUxcTE2LazsrIUEhKi6OhoNWzYsAq+CwA1ydVquaFDh6qgoKCa0gBwV6VXvFcGjTUAqEBsbKwWLFigoqIiSVJSUpKef/55Pf3005o7d67J6QCgdggODlanTp3sxjp27Ki1a9dKkoKCgiRJp0+fVnBwsG3OmTNnyl3FVspqtcpqtZYb9/LykpeXl6uiA6iBtmzZYnu9f/9+hYWFKSkpSYMHD9bhw4fVsWNHSZeesdavXz+zYgJwA47UDCxeAAC/Exsbq3nz5qlp06ZaunSpVq5cqaVLl6pp06aaN2+eYmNjzY4IALVC3759dfDgQbux7777Tm3atJEkhYaGKigoSBs3brTtLygoUGpqqvr06VOtWQHUfP3797e9Dg8Pt9tXdrvsPAC4GhprAFBGQUGBFixYoMDAQB0/flxjxoxR48aNNWbMGB0/flyBgYFasGABtwgAgAs8/fTT2rZtm+Lj43Xo0CElJiZq2bJlGj9+vKRLt4BOnjxZ8fHx+uijj7R3716NHj1avr6+GjFihMnpAdRUDz30UIXjw4cPr+YkAGoDGmsAUMbixYtVVFSkWbNmydPT/m55T09PvfzyyyoqKtLixYtNSggAtUevXr300UcfafXq1ercubNeeeUVLVy40O5Nb2xsrCZPnqxx48apZ8+eOnHihJKTk+Xv729icgA12XvvvVfheOlt6ADgCJ6xBgBlHD58WJJ01113Vbi/dLx0HgDgj7nrrrsue86VLl21FhcXp7i4uOoLBaBWSktLs93muWDBAruFTubPn283DwAqiyvWAKCMsLAwSdKnn35a4f7S8dJ5AAAAqBnKLkhQtqn2+20WLgDgCBprAFDGuHHj5OnpqRdffNG2ImipoqIivfTSS/L09NS4ceNMSggAAAAAcBdO3Qqal5enb775RmfOnFFJSYndvv/8z/90STAAMEP9+vX19NNPa968eWrVqpVmzJghb29v/fd//7dmzpypn376SVOmTFH9+vXNjgoAptm+fbtSUlIqrAXL3k4FAO7kk08+sb1etGiRbaGU329/8sknvK8FUGkON9Y+++wzPfzww/rll1/K7bNYLCouLnZJMAAwy9y5cyVdevZG2SvTPD09NWXKFNt+AKiL4uPj9eKLL6pDhw4KDAyUxWKx7Sv7GgDczZAhQ2yvjxw5Yrev7PaQIUNkGEa15QJQs1kMB88Y7dq108CBA/XSSy8pMDCwqnJVq6ysLAUEBOj8+fNq2LCh2XEAuInc3FzFxMRo27ZtuvnmmzV//nz5+PiYHQuAm6ir9UNgYKDmzJmj0aNHmx2lUurq7wlAeY40/2msAXWbI/WDw89YO3PmjGJiYmpNUw0AKvLhhx+qU6dOWrp0qXbt2qWlS5eqU6dO+vDDD82OBgCmqlevnvr27Wt2DABwCQ8PD7MjAKjhHG6s3XvvvUpJSamCKADgHj788EPde++96tKli9LS0rR69WqlpaWpS5cuuvfee2muAajTnn76aS1atMjsGADgsDvvvNP2etKkSSooKNDatWtVUFCgSZMmVTgPAK7G4VtBc3JydN999+maa65Rly5d5OXlZbd/4sSJLg1YHbhFAECp4uJitWvXTl26dNG6detUXFyspKQkDR48WB4eHho6dKj27t2r77//nk84gTqurtYPJSUluvPOO/Xdd9+pU6dO5WpBd/vwoa7+ngCUV9GtoO3bt9f3339fbpxbQYG6zZH6weHFCxITE7Vhwwb5+PgoJSWl3ANra2JjDQBKpaWl6ccff9Tq1atVr149uwVZ6tWrp6lTp6pPnz5KS0tTZGSkeUEBwCQTJkzQpk2bFBUVpaZNm7JgAYAaraKmGgA4wuHG2osvvqiXX35Zzz//vOrVc/hOUgBwa6dOnZIkde7cucL9peOl8wCgrnnnnXe0du1abpUCUKN9/PHHdquE/n4bACrL4c5YQUGBHnjgAZpqAGql4OBgSdLevXsr3F86XjoPAOqaJk2aKCwszOwYAOCw2NhY2+s5c+bY7Su7XXYeAFyNw92xRx55RH//+9+rIgsAmK5///5q27at4uPjVVJSYrevpKRECQkJCg0NVf/+/U1KCADmiouL04wZM5STk2N2FABwSNnm2datW+32ld3+fdMNAK7E4VtBi4uLNXfuXG3YsEFdu3Yt98Da+fPnuywcAFQ3Dw8Pvf7667r33ns1dOhQTZkyRbm5udq2bZvmzZunTz/9VB988AELFwCos958800dPnxYgYGBatu2bblacOfOnSYlAwAAqH4ON9b27Nmj7t27Syp/q5SrH15bVFSkuLg4vffeezp9+rSCg4M1evRovfjii7ZbUQ3D0MyZM7Vs2TKdO3dOvXv31qJFi3T99de7NAuAumPYsGH64IMP9Mwzz+jWW2+1jYeGhuqDDz7QsGHDTEwHAOYaOnSo2REAwCmTJk2yve7du7e++uqrCrcnTZqkN954o9rzAaiZLIYbryM8e/ZsLViwQKtWrdL111+vHTt26NFHH9WsWbNsJ8U5c+Zo9uzZ+tvf/qbrrrtOs2bN0ubNm3Xw4EH5+/tX6t9hGXYAFSkuLtamTZu0fv16DRo0SFFRUVypBsCG+qFm4PcEoFTZC0EMw1BhYaGSkpI0ePBgeXl5ldsPoO5ypH5w+Iq16vTll19qyJAhtlWn2rZtq9WrV2vHjh2SLp3sFi5cqGnTptmuIFm1apUCAwOVmJiosWPHmpYdQM3n4eGhiIgIZWdnKyIigqYaAJSxY8cO7d+/XxaLRR07dlSPHj3MjgQAlVK/fv0Kxz08PFRcXFzNaQDUdE411tLT0/X+++/r6NGjKigosNv34YcfuiSYJPXr109Lly7Vd999p+uuu067d+/Wli1btHDhQklSZmamTp8+rejoaNvXWK1WRUREaOvWrZdtrOXn5ys/P9+2nZWVJUkqLCxUYWGhy/IDqPlKzwmcGwD8Xl09Lxw/flx/+tOf9MUXX6hRo0aSpF9//VV9+vTR6tWrFRISYm5AALiK37+HLUVTDYAzHG6srVmzRg8//LCio6O1ceNGRUdH6/vvv9fp06d1zz33uDTcc889p/Pnzys8PNz26cHs2bP1pz/9SZJ0+vRpSVJgYKDd1wUGBurIkSOXPW5CQoJmzpxZbjw5OVm+vr4u/A4A1BYbN240OwIAN1NXV8UcM2aMCgsLtX//fnXo0EGSdPDgQY0ZM0aPPfaYkpOTTU4IABWbOHGi3nzzTUlSUFCQfvrpJ9u+su8pJ06cWO3ZANRcDj9jrWvXrho7dqzGjx8vf39/7d69W6GhoRo7dqyCg4MrbFg5a82aNZoyZYrmzZun66+/Xrt27dLkyZM1f/58PfLII9q6dav69u2rkydPKjg42PZ1jz/+uI4dO6bPPvuswuNWdMVaSEiIfvnlF569AcBOYWGhNm7cqAEDBpRb+Q5A3ZaVlaVmzZrVuWd3+fj4aOvWrbbFrErt3LlTffv2VW5urknJKsYz1gCUVZkF93i+GoAqfcba4cOHbc88s1qtys7OlsVi0dNPP63bbrvNpY21KVOm6Pnnn9eDDz4oSerSpYuOHDmihIQEPfLIIwoKCpIk24qhpc6cOVPuKrayrFarrFZruXEvLy/eOAOoEOcHAL9XV88JrVu3rvA22KKiIrVs2dKERAAAAOap5+gXNGnSRBcuXJAktWzZUnv37pV06dkarr4lIicnR/Xq2Uf08PBQSUmJJCk0NFRBQUF2t2gVFBQoNTVVffr0cWkWAAAASHPnztWECRO0Y8cO21UdO3bs0KRJk/Taa6+ZnA4ALu+OO+6wvf798yDLbpedBwBX4/AVa/3799fGjRvVpUsX3X///Zo0aZL+9a9/aePGjbr99ttdGu7uu+/W7Nmz1bp1a11//fX6+uuvNX/+fI0ZM0bSpct4J0+erPj4eLVv317t27dXfHy8fH19NWLECJdmAQAAgDR69Gjl5OSod+/e8vS8VEoWFRXJ09NTY8aMsdVpknT27FmzYgJAORs2bLC9Pnr0qAoLC5WUlKTBgwfLy8vLdpto2XkAcDUON9befvtt5eXlSZKmTp0qLy8vbdmyRcOGDdP06dNdGu6tt97S9OnTNW7cOJ05c0YtWrTQ2LFj9dJLL9nmxMbGKjc3V+PGjdO5c+fUu3dvJScny9/f36VZAAAAINvq7AAAAHBi8YLaiIfaAric33+SCQClqB9qBn5PAEqVXbjAMIzLXrFWuh9A3eVI/eDwM9ays7O1efNm/f3vf9cHH3ygjIwMTjoAAAB11E8//aSjR4+aHQMArmrgwIG210899ZTdvrLbZecBwNVUurFWUlKi2NhYNW/eXFFRURoxYoTuv/9+9erVS6GhofrHP/5RlTkBAABgogsXLmjkyJFq06aNHnnkERUUFGj8+PEKDg5WaGioIiIilJWVZXZMALiszz77zPZ60aJFql+/voYOHar69etr0aJFFc4DgKupdGPthRde0KeffqrExEQlJSWpb9++evXVV/Xtt9/q4Ycf1n333afk5OSqzAoA1erixYsaPny4Jk2apOHDh+vixYtmRwIA07zwwgvKyMjQs88+q6NHj+r+++/X5s2blZaWppSUFJ09e1Zz5swxOyYAXNHV7rbibiwAjqr0M9ZatmypNWvWqH///pKkEydOKDw8XL/88ousVqteeeUVrV+/Xlu3bq3SwFWBZ28A+L2bbrpJ6enp5cZ79eql7du3m5AIgLupa/VD69attWrVKkVFRenkyZNq1aqVPv74Y919992SpKSkJMXExOjAgQMmJ7VX135PAK6s7HPULofmGoAqecbahQsX1LJlS9t2cHCw8vLydO7cOUnS8OHDtXv3bicjA4D7KG2qWSwWjRw5UgsWLNDIkSNlsViUnp6um266yeyIAFDtzpw5o3bt2kmSWrRoIR8fH3Xo0MG2//rrr9exY8fMigcAV1W2qebn56eCggKtW7dOBQUF8vPzq3AeAFxNpRtrXbp00erVq23b//u//ys/Pz8FBQVJuvQMNqvV6vqEAFCNLl68aGuq5eTkaMWKFQoNDdWKFSuUk5Nja65xWyiAuqZp06b6+eefbdtDhgxRo0aNbNsXL16kFgRQY1y4cOGK2wBQWZVurL388st65ZVX1Lt3b0VERGjUqFGaMWOGbf9nn32m7t27V0lIAKguo0aNkiSNHDlS3t7edvu8vb01YsQIu3kAUFd07drV7hb5xMRENW/e3Ladnp6ujh07mhENAADANJVurN1+++3avn27/uM//kO9evVSUlKSJk+ebNv/7LPP6vPPP6+KjABQbQ4fPizp0jmtuLhYqamp2rx5s1JTU1VcXKyYmBi7eQBQV7z33nt64IEHLrs/MDBQs2fPrsZEAAAA5vN0ZHLXrl3VtWvXqsoCAKYLCwvTnj17NHHiRB05ckQ//vijJGn+/Plq27atQkJCbPMAoC5p0qTJFfcPGjSompIAwB/Hc9QAuEqlGmvffPNNpQ9I4w1ATfbuu+/K399fqampuvPOO/Xuu+/q+PHjatWqlV599VX985//tM0DgLqCWhBAbWAYBquCAnC5SjXWunXrJovFYjvBXOlkVFxc7JpkAGACHx8f1a9fXwUFBfrnP/+pgIAA9ejRQ//4xz9sTTWr1SofHx+TkwJA9SlbC17tTSm1IAAAqEsq9Yy1zMxM/fDDD8rMzNSHH36o0NBQLV68WF9//bW+/vprLV68WGFhYVq7dm1V5wWAKpWWlqaCggLbA7gTExP1zDPPKDExUZLUsWNH5efnKy0tzcyYAFCtytaCa9eupRYEUCNV9vZPbhMF4IhKXbHWpk0b2+v77rtPb775pgYPHmwb69q1q0JCQjR9+nQNHTrU5SEBoLqcOnVKkrR9+3ZJ0ogRI/TNN9+oa9euSkxMlGEYatiwoW0eANQF1IIAahvDMFRYWKikpCQNHjxYXl5eNNQAOKXSq4KW2rNnj0JDQ8uNh4aG6ttvv3VJKAAwS3BwsCRp79698vPz09q1a/XGG29o7dq18vPz0969e+3mAUBdQy0IAADwG4cbax07dtSsWbOUl5dnG8vPz9esWbNst04BQE3Vv39/tW3bVvHx8crLy9Obb76pZcuW6c0331ReXp4SEhIUGhqq/v37mx0VAExBLQgAAPCbSt0KWtbSpUt19913KyQkRDfccIMkaffu3bJYLPr0009dHhAAqpOHh4def/11DR8+XL6+vrZFW5KSkjRlyhQZhqG1a9fKw8PD5KQAYA5qQQC1Abd9AnAVhxtrN910kzIzM/U///M/OnDggAzD0AMPPKARI0aoQYMGVZERAKrVtm3bJMluNWRJqlevnoqLi7Vt2zYNGzbMrHgAYCpqQQA1VWVWNi6dBwCVZTE4aygrK0sBAQE6f/68GjZsaHYcACYqKChQgwYN1LRpUx05ckRpaWlav369Bg0apP79+6tNmzb697//rezsbNWvX9/suABMRP1QM/B7AlAWjTUAleFI/eDwM9Yk6d1331W/fv3UokULHTlyRJK0YMECffzxx84cDgDcxuLFi1VUVKRZs2bJarUqIiJCt956qyIiImS1WvXyyy+rqKhIixcvNjsqAJiGWhBATVTZ2z+5TRSAIxxurC1ZskQxMTEaNGiQzp07p+LiYklS48aNtXDhQlfnA4BqdfjwYUnSXXfdpdzcXE2cOFFxcXGaOHGicnNzddddd9nNA4C6hloQQG1gGIYKCgq0bt06FRQUcJUaAKc53Fh76623tHz5ck2bNk2enr89oq1nz57as2ePS8MBQHULCwuTJA0aNEi+vr5aunSpdu3apaVLl8rX11eDBw+2mwcAdQ21IAAAwG8cbqxlZmaqe/fu5catVquys7NdEgoAzDJu3DhJ0q5du1S/fn3FxsZqyZIlio2NVf369bV79267eQBQ11ALAgAA/MbhVUFDQ0O1a9cutWnTxm58/fr16tSpk8uCAYAZSm9pkqSAgAC1bdtWVqtVbdu2VUBAgH7++edy8wCgLqEWBFAb8Bw1AK7icGNtypQpGj9+vPLy8mQYhrZv367Vq1crISFB//3f/10VGQGg2kyZMkWS1Lt3b2VkZNhdmebp6alevXopPT1dU6ZM0dtvv21WTAAwDbUggJrKMAxWBQXgcg431h599FEVFRUpNjZWOTk5GjFihFq2bKk33nhDDz74YFVkBIBq8/3330uS3nvvPXl5een6669Xdna2GjRooH379ikvL08dOnSwzQOAuoZaEAAA4DcON9Yk6fHHH9fjjz+uX375RSUlJWrevLmrcwGAKdq3b6/k5GSFh4erqKjINn7x4kW1adPG9qDu9u3bmxURAExHLQigJqrs7Z8Wi4Wr1gBUmsOLF0hSUVGR/t//+39au3atfHx8JEknT57UxYsXXRoOAKrbvHnzJMnWVGvbtq2effZZtW3b1m68dB4A1EXUggAAAJc4fMXakSNHdMcdd+jo0aPKz8/XgAED5O/vr7lz5yovL09Lly6tipwAUC3Onz9ve+3p6al7771XoaGhuvfee7Vw4UJbY+38+fO2N5MAUJdQCwIAAPzG4cbapEmT1LNnT+3evVtNmza1jd9zzz3685//7NJwAFDdunXrJkny8fFRbm6uXnvtNbv9pePdunXT6dOnTUgIAOaiFgQAAPiNw7eCbtmyRS+++KLq169vN96mTRudOHHCZcEAwAy//vqrJOmdd97RDz/8IG9vb0mSt7e3fvjhB9uKd6XzAKCuoRYEUFsUFBRo3bp1KigoMDsKgBrM4SvWSkpKVFxcXG78+PHj8vf3d0koADBLo0aN9NNPP+m+++6zG8/Ly9O1115rNw8A6iJqQQC1xe8/IAAAZzh8xdqAAQO0cOFC27bFYtHFixc1Y8YMDR482JXZAKDa7dq1y27794sXXG4eANQV1IIAAAC/cfiKtQULFigqKkqdOnVSXl6eRowYoe+//17NmjXT6tWrqyIjAJjGMAwVFRWx5DoA/B9qQQAAgN9YDCfeLebm5mr16tXauXOnSkpKdOONN+qhhx6qsSvkZWVlKSAgQOfPn1fDhg3NjgPAREFBQfrpp5+uOi8wMJDFC4A6ri7XDzWpFqzLvycA9iwWS6Xn8qEqULc5Uj841VirbSi4AJTy9vZWfn6+3n//fbVs2VJ9+vSx7du6dasyMzP10EMPyWq1Ki8vz8SkAMxG/VAz8HsCUFZlmmu8RQbgSP3g8K2gknTw4EG99dZb2r9/vywWi8LDw/XUU08pPDzcqcAA4C4ut3iBJLsmG4sXAKjLqAUBAAAucXjxgg8++ECdO3dWRkaGbrjhBnXt2lU7d+5Uly5d9P7771dFRgCoNr9flKBhw4Z67LHHyn1KweIFAOoqakEANVVlbwV15JZRAHD4irXY2FhNnTpVL7/8st34jBkz9Nxzz1V4lQcA1BS5ubl2240bN5a/v78aN26srKysy84DgLqCWhBAbWAYhgoLC5WUlKTBgwfLy8uLhhoApzh8xdrp06f18MMPlxsfOXJklTzI+8SJExo5cqSaNm0qX19fdevWTRkZGbb9hmEoLi5OLVq0kI+PjyIjI7Vv3z6X5wBQN3Tq1Mlu+8iRI1q4cKGOHDlyxXkAUFdUVS2YkJAgi8WiyZMn28ao8wAAgLtzuLEWGRmptLS0cuNbtmxR//79XRKq1Llz59S3b195eXlp/fr1+vbbb/X666/bPdto7ty5mj9/vt5++22lp6crKChIAwYM0IULF1yaBUDdkJ+fL0l68803deDAAXl4eEiSPDw8dODAAb322mt28wCgrqmKWjA9PV3Lli1T165d7cap8wAAgLtz+FbQ//zP/9Rzzz2njIwM3XzzzZKkbdu26f3339fMmTP1ySef2M39I+bMmaOQkBCtXLnSNta2bVvba8MwtHDhQk2bNk3Dhg2TJK1atUqBgYFKTEzU2LFj/9C/D6DuKV3tc+LEiXbjxcXFdg/ltlqt1R0NANyCq2vBixcv6qGHHtLy5cs1a9Ys2zh1HoCqZLFYVFBQYLcNAM6wGA6uJVyvXuUucrNYLCouLnYqVKlOnTpp4MCBOn78uFJTU9WyZUuNGzdOjz/+uCTphx9+UFhYmHbu3Knu3bvbvm7IkCFq1KiRVq1aVeFx8/Pz7a42ycrKUkhIiH755ReWYQfquMzMTHXo0MG2HR0drdtuu03/+te/lJycbBs/ePCgQkNDzYgIwE1kZWWpWbNmlVqGvTZxdS34yCOPqEmTJlqwYIEiIyPVrVs3LVy4kDoPQJWoX7/+VeeUbbgBqJscqfMcvmKtpKTE6WCO+uGHH7RkyRLFxMTohRde0Pbt2zVx4kRZrVY9/PDDtud4BAYG2n1dYGBguechlZWQkKCZM2eWG09OTpavr69rvwkANcrFixftttPS0hQUFFTutqf09HTt37+/OqMBcDM5OTlmRzCFK2vBNWvWaOfOnUpPTy+3jzoPQFVYt26dhg4desX9SUlJ1RcIgFtypM5zuLFWnUpKStSzZ0/Fx8dLkrp37659+/ZpyZIldg/N/f1lu4ZhXPFS3qlTpyomJsa2XfpJZnR0NJ9kAnVcRESE3XZubq7eeeedcvMWLVqk1NTU6ooFwA2VXSkYjjt27JgmTZqk5ORkeXt7X3YedR4AVysoKKjwyjWuVANQypE6r9KNta+++kpnz57VoEGDbGPvvPOOZsyYoezsbA0dOlRvvfWWS587FBwcXG7lvY4dO2rt2rWSpKCgIEmXPtEMDg62zTlz5ky5TzfLslqtFeb08vKSl5eXK6IDqKGOHz8uSfryyy+VmZmpESNG2PYlJiaqVatWuvXWW3X8+HHOF0AdV9fOAa6uBTMyMnTmzBn16NHDNlZcXKzNmzfr7bff1sGDByVR5wGoGoZhqLCwUElJSRo8eDDnBwB2HDknVLqxFhcXp8jISFsxtWfPHj322GMaPXq0OnbsqHnz5qlFixaKi4tzOPDl9O3b11ZUlfruu+/Upk0bSVJoaKiCgoK0ceNG27M3CgoKlJqaqjlz5rgsB4C6o3Xr1jp27JhuueWWcvvKNtlat25dnbEAwHSurgVvv/127dmzx27s0UcfVXh4uJ577jlde+211HkA7OTk5OjAgQMuO97F3Hxt3XNYjZvtkJ+P6y4QCQ8P59ZzoA6pdGNt165deuWVV2zba9asUe/evbV8+XJJUkhIiGbMmOHSxtrTTz+tPn36KD4+Xvfff7+2b9+uZcuWadmyZZIu3RowefJkxcfHq3379mrfvr3i4+Pl6+tr9wYYACrrn//8pxo1alSpeQBQl7i6FvT391fnzp3txho0aKCmTZvaxqnzAJR14MABu6tcXWWui4+XkZGhG2+80cVHBeCuKt1YO3funN1l96mpqbrjjjts27169dKxY8dcGq5Xr1766KOPNHXqVL388ssKDQ3VwoUL9dBDD9nmxMbGKjc3V+PGjdO5c+fUu3dvJScny9/f36VZANQN3377baXnVXRVGwDUVmbUgtR5AMoKDw9XRkaGy4538NSvinl/j+bf10Udghu57Ljh4eEuOxYA92cxDMOozMQ2bdro3Xff1a233qqCggI1atRI//jHP3T77bdLunQ7QEREhM6ePVulgatCVlaWAgICKrWMKoDa7UoPxP69Sp4+AdRSda1+qKm1YF37PQGovF1H/q2hS7Zp3ZM3q1ubpmbHAeBGHKkf6lX2oHfccYeef/55paWlaerUqfL19VX//v1t+7/55huFhYU5nxoA3Mxf//pX3XLLLWrWrJluueUW/fWvfzU7EgCYhloQAACgvErfCjpr1iwNGzZMERER8vPz06pVq+yWKF6xYoWio6OrJCQAmGHMmDEaNWqU3WpRjz32mNmxAMAU1IIAAADlVbqxds011ygtLU3nz5+Xn5+fPDw87Pa///778vPzc3lAADDLtGnTFB8fb9t+4YUXTEwDAOaiFgQAACiv0o21UgEBARWON2nS5A+HAQB3UrapVtE2ANRF1IIAAAC/qVRjbdiwYZU+4Icffuh0GAAw2/z58xUTE1OpeQBQV1ALAgAAVKxSixcEBATY/jRs2FCff/65duzYYdufkZGhzz///LKfYAJATeHr6+vSeQBQG1ALAgAAVKxSV6ytXLnS9vq5557T/fffr6VLl9qerVFcXKxx48axhDmAGu+JJ56o9LyxY8dWcRoAcA/UggAAABWr1BVrZa1YsULPPvus3QNrPTw8FBMToxUrVrg0HACY5dprr1WPHj3sxnr06KGQkBCTEgGAe6AWBAAA+I3DixcUFRVp//796tChg934/v37VVJS4rJgAGCmH374odxYRkaGCUkAwL1QCwIAAPzG4cbao48+qjFjxujQoUO6+eabJUnbtm3Tq6++qkcffdTlAQGgOi1durRSt4MuXbq0GtIAgPuhFgQAAPiNw4211157TUFBQVqwYIFOnTolSQoODlZsbKyeeeYZlwcEgOq0c+fOcmONGzfWuXPnrjoPAOoCakEAAIDfWAzDMJz94qysLEmq8Q+qzcrKUkBAgM6fP1/jvxcAf4zFYqn03D9w+gRQC1A/1IxakN8TgMvZdeTfGrpkm9Y9ebO6tWlqdhwAbsSR+sHhxQvKatiwIQUKgFqpXr16Cg0NtRv7/TYA1HXUggAAoK5zuLH2008/adSoUWrRooU8PT3l4eFh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7rhDp06dsn1tcXGxYmJi1KhRIzVt2lSxsbEyDMO2/9dff9WWLVs0Z84cRUVFqU2bNrrppps0depU3XnnnbZ5FotFS5Ys0aBBg+Tj46PQ0FC9//77dt/TiRMn9MADD6hx48Zq2rSphgwZoh9//NFuzsqVK9WxY0d5e3srPDxcixcvttu/fft2de/eXd7e3urZs6e+/vpru/1r1qxR165d9dJLL6ldu3aKiIhQQkKCFi1apAsXLki69Gnnn/70J7Vq1Uq+vr7q0qWLVq9ebXecyMhITZgwQZMnT1bjxo0VGBioZcuWKTs7W48++qj8/f0VFham9evXX+U3CQAAYI86z7k6r3HjxnryySfVs2dPtWnTRrfffrvGjRuntLQ025zSn8u6det03XXXydvbWwMGDNCxY8dsc+Li4tStWzetWLFCrVu3lp+fn5588kkVFxdr7ty5CgoKUvPmzTV79uyr/CYBuBqNNQAuFRkZqU2bNtm2N23apMjISEVERNjGCwoK9OWXXyoqKqrCY+Tk5Oi1117Tu+++q82bN+vo0aN69tlnbftff/11rVixQn/961+1ZcsWnT17Vh999JFtv5+fn/z8/LRu3Trl5+dfMe/06dM1fPhw7d69WyNHjtSf/vQn7d+/35YjKipKfn5+2rx5s7Zs2WIrAAsKCiRJy5cv17Rp0zR79mzt379f8fHxmj59ulatWiVJys7O1l133aUOHTooIyNDcXFxdt+LJOXn58vb29tuzMfHR3l5ecrIyJAk5eXlqUePHvr000+1d+9e/eUvf9GoUaP01Vdf2X3dqlWr1KxZM23fvl0TJkzQk08+qfvuu099+vTRzp07NXDgQI0aNUo5OTlX/LkAAAD8HnWe43Xe7508eVIffvihIiIiyv1cZs+erVWrVumLL75QVlaWHnzwQbs5hw8f1vr16/XZZ59p9erVWrFihe68804dP35cqampmjNnjl588UVt27btihkAuJgBAC60bNkyo0GDBkZhYaGRlZVleHp6Gj/99JOxZs0ao0+fPoZhGEZqaqohyTh8+LCxadMmQ5Jx7tw5wzAMY+XKlYYk49ChQ7ZjLlq0yAgMDLRtBwcHG6+++qptu7Cw0GjVqpUxZMgQ29gHH3xgNG7c2PD29jb69OljTJ061di9e7ddVknGE088YTfWu3dv48knnzQMwzD++te/Gh06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"/>

특성의 분포 확인



히스토그램으로 정규 분포인지 왜곡된 분포인지 확인한다.



정규 분포를 따를 경우 극값 분석을, 왜곡된 분포일 경우 IQR을 찾는다.



```python
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
```

<pre>
Text(0.5, 0, 'WindSpeed3pm')
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABPEAAANBCAYAAACBMCtjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAADqS0lEQVR4nOzdf1yUdb7//+eIMAKrE0iAU1i2a6RhrQc3Rfez6iqgR2Q7nl0ratLNJXctWRKzzG0XK7HMXy2crMyTruiye/uYnVKXBdvS5YM/SU6hHu2U648CaRPBXw0TXN8/5sulI/4CBxjgcb/duN2c63rNNe/30ynfvOaa67IYhmEIAAAAAAAAgM/q0tYDAAAAAAAAAHBlNPEAAAAAAAAAH0cTDwAAAAAAAPBxNPEAAAAAAAAAH0cTDwAAAAAAAPBxNPEAAAAAAAAAH0cTDwAAAAAAAPBxNPEAAAAAAAAAH9e1rQfQ2dTX1+vLL79U9+7dZbFY2no4AACgHTAMQ6dOnZLdbleXLnwG66tY5wEAgKZqyjqPJl4r+/LLLxUVFdXWwwAAAO3Q0aNHdfPNN7f1MHAZrPMAAEBzXcs6jyZeK+vevbsk919Ojx49vH58l8ulgoICJSQkyN/f3+vH93Wdff4SGUhkIJFBZ5+/RAZSx8qgpqZGUVFR5joCvol1XusgBzdycCOH88jCjRzcyMGtPeTQlHUeTbxW1vDVih49erTY4i4oKEg9evTw2TdoS+rs85fIQCIDiQw6+/wlMpA6ZgZ8RdO3sc5rHeTgRg5u5HAeWbiRgxs5uLWnHK5lncdFVQAAAAAAAAAfRxMPAAAAAAAA8HE08QAAAAAAAAAfRxMPAAAAAAAA8HE08QAAAAAAAAAfRxMPAAAAAAAA8HE08QAAANAmvv32W/3mN79Rnz59FBgYqNtuu03PPfec6uvrzRrDMJSZmSm73a7AwECNGDFCe/fu9TiO0+nU9OnTFRYWpuDgYCUnJ+vYsWMeNVVVVXI4HLLZbLLZbHI4HDp58qRHzZEjRzR+/HgFBwcrLCxMaWlpqq2tbbH5AwAANAVNPAAAALSJl156Sa+99ppycnK0f/9+LViwQC+//LKys7PNmgULFmjx4sXKycnRrl27FBkZqfj4eJ06dcqsSU9P1/r165WXl6eioiKdPn1aSUlJqqurM2tSUlJUWlqq/Px85efnq7S0VA6Hw9xfV1encePG6cyZMyoqKlJeXp7WrVunjIyM1gkDAADgKtq0ibd161aNHz9edrtdFotF77zzzmVrp06dKovFoqVLl3psb81PXj/55BMNHz5cgYGBuummm/Tcc8/JMIzriQAAAKDT2rZtm37yk59o3LhxuvXWW/XTn/5UCQkJ2r17tyT3WXhLly7VnDlzNGHCBMXExGjVqlU6e/as1q5dK0mqrq7WihUrtGjRIo0ePVoDBw5Ubm6uPvnkE23evFmStH//fuXn5+vNN99UXFyc4uLitHz5cm3YsEEHDhyQJBUUFGjfvn3Kzc3VwIEDNXr0aC1atEjLly9XTU1N2wQEAABwga5t+eJnzpzR3XffrZ///Of693//98vWvfPOO9qxY4fsdnujfenp6XrvvfeUl5ennj17KiMjQ0lJSSopKZGfn58k9yevx44dU35+viTp0UcflcPh0HvvvSfp/CevN954o4qKivT1119r0qRJMgzD/CS4pqZG8fHxGjlypHbt2qWDBw9q8uTJCg4O5hNaAACAZvjhD3+o1157TQcPHtTtt9+u//7v/1ZRUZH5oe2hQ4dUUVGhhIQE8zlWq1XDhw9XcXGxpk6dqpKSErlcLo8au92umJgYFRcXKzExUdu2bZPNZtPgwYPNmiFDhshms6m4uFjR0dHatm2bYmJiPNabiYmJcjqdKikp0ciRIxuN3+l0yul0mo8bmn0ul0sul8trOTVoOGZLHLs9IQc3cnAjh/PIwo0c3MjBrT3k0JSxtWkTb+zYsRo7duwVa7744gs9/vjj+utf/6px48Z57Gv45HX16tUaPXq0JCk3N1dRUVHavHmzEhMTzU9et2/fbi7cli9frri4OB04cEDR0dHmJ69Hjx41F26LFi3S5MmTNW/ePPXo0UNr1qzRN998o5UrV8pqtSomJkYHDx7U4sWLNWPGDFkslhZICAAAoON66qmnVF1drTvuuEN+fn6qq6vTvHnz9MADD0iSKioqJEkREREez4uIiNDhw4fNmoCAAIWEhDSqaXh+RUWFwsPDG71+eHi4R83FrxMSEqKAgACz5mLz58/X3LlzG20vKChQUFDQVeffXIWFhS127PaEHNzIwY0cziMLN3JwIwc3X87h7Nmz11zbpk28q6mvr5fD4dCTTz6pO++8s9H+1vzkddu2bRo+fLisVqtHzezZ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dEtPHwAAwGc1rNMu/AkJCdEDDzyg+++/36PW5XIpLCzMbPrl5+frhz/8oW644Qb17NlTSUlJ+uyzz8z6f/zjH7JYLMrLy9PQoUPVrVs33Xnnnfrwww89jrtlyxbdc889slqt6tWrl55++ml9++235v4RI0bo8ccf14wZMxQWFqb4+HhJV17bffjhh/r5z3+u6upq8wzDhvXixV+nPXLkiH7yk5/oO9/5jnr06KGJEyfq+PHj5v6GS6ysXr1at956q2w2m+6//36dOnXquvMH0PJo4gFAM33++efKz8+Xv7+/JKm+vl4333yz/vznP2vfvn367W9/q2eeeUZ//vOfr3icVatWKTg4WDt27NCCBQv03HPPqbCwUJL7KxJjxozRxIkTVV5ebjYFu3Tpot///vcqKyvTqlWr9Le//U2zZs1q2QkDAAC0Qw8++KDeffddjw88//rXv+rMmTP693//d0nSmTNnNGPGDO3atUvvv/++unTpon/7t39TfX29x7GefPJJZWRkaM+ePRo6dKiSk5P19ddfS5K++OIL/eu//qt+8IMf6L//+7+1bNkyrVixQi+88ILHMVatWqWuXbvq//2//6fXX39d0pXXdkOHDtXSpUvVo0cPlZeXq7y8XDNnzmw0T8MwdO+99+rEiRPasmWLCgsL9dlnn+m+++7zqPvss8/0zjvvaMOGDdqwYYO2bNmiF1988TpTBtAa+DotADTBhg0b9J3vfEd1dXX65ptvJLk/OZUkf39/zZ0716zt06ePiouL9ec//1kTJ0687DHvuusu/e53v5Mk9e3bVzk5OXr//fcVHx+vG2+8UVarVYGBgYqMjDSfk56e7vE6zz//vH71q1/p1Vdf9eZ0AQAA2o2GddqFnnrqKT399NMKDg7W+vXr5XA4JElr167V+PHj1aNHD0kym3kNVqxYofDwcO3bt08xMTHm9scff9ysXbZsmfLz87VixQrNmjVLr776qqKiopSTkyOLxaI77rhDX375pZ566in99re/VZcu7nNovve972nBggUer3eltV1AQIBsNpssFovHevBimzdv1scff6xDhw4pKipKkrR69Wrdeeed2rVrl37wgx9Icn/wvHLlSnXv3l2S5HA49P7772vevHnXFjSANkMTDwCaYOTIkVq2bJnOnj2rN998UwcPHtT06dPN/a+99prefPNNHT58WOfOnVNtbe1V7wp71113eTzu1auXKisrr/icDz74QFlZWdq3b59qamr07bff6ptvvtGZM2cUHBzc7PkBAAC0Vw3rtAuFhobK399fP/vZz7RmzRo5HA6dOXNG//Vf/6W1a9eadZ999pmeffZZbd++Xf/85z/NM/COHDni0cSLi4sz/9y1a1cNGjRI+/fvlyTt379fcXFxslgsZs2wYcN0+vRpHTt2TL1795YkDRo0qNHYvbG2279/v6KioswGniT1799fN9xwg/bv32828W699VazgSdd29oTgG/g67QA0ATBwcH63ve+p7vuuku///3v5XQ6zbPv/vznP+uJJ57QI488ooKCApWWlurnP/+5amtrr3jMhq/jNrBYLI2+unGhw4cP61//9V8VExOjdevWqaSkRP/xH/8hyX19FwAAgM6oYZ124U9oaKgk91dqN2/erMrKSr3zzjvq1q2bxo4daz53/Pjx+vrrr7V8+XLt2LFDO3bskKSrruMkmU07wzA8GngN2y6saRjnhby1trvU619qe1PXngB8B008ALgOv/vd77Rw4UJ9+eWX+vvf//7/tXf/cVHW+f7/nyM/RiCcQIKRwrIy0kXdFk+ItouuAnZE6rhnbaMod80sSiP1Y5nbhlti+XtXdis9lq4/ls4es23VCGzL4uBPlN1Qs7YstUAqEfw5jHB9//DLdRpRG5VxBuZxv924rdf7es37/b5e19Be8+J9zaX+/fsrJydHt9xyi2688UaXL0RuLdu2bdOpU6c0Z84c9evXTzfddJO++uqrVh8HAACgvejfv7/i4uL02muvacWKFfr5z39uPhX222+/1e7du/XrX/9agwcPVo8ePVRbW3vWfjZt2mT++9SpUyovL9fNN98s6fSqt7KyMrNwJ0llZWUKDw/X1Vdffc65uXNtFxwcrMbGxvMeY8+ePbVv3z7t37/fbNu1a5fq6urUo0eP874WQNtAEQ8ALsHAgQP1gx/8QPn5+brxxhu1bds2vf322/r444/19NNPa+vWra0+5g033KBTp05pwYIF+uyzz7Rs2TK99NJLrT4OAABAW+JwOFRdXe3y880330g6vdosKytLL730kkpKSnTvvfear4uIiFDnzp21cOFC/etf/9Lf//53TZgw4axj/OEPf9Dq1av10Ucf6ZFHHlFtba1+9atfSZJycnK0f/9+jRs3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9eijj+qf//ynSktLW+zz57zt379fjz32mIqLi9WxY8dzxvlzjiSpqalJffv2VX5+viTplltu0c6dO/Xiiy/qvvvuM+P8PU+vvfaali9frpUrV+oHP/iBKioqlJubq9jYWN1///1mnL/n6WwuJif+mjen06lf/OIXampq0h//+MfvjffXPKH94b+dbQfXnW0H18JtF9fnbROfFy4dt9O2A1FRUQoICGjx19iampoWFWx/NG7cOL355pt69913dc0115jtdrtdkvw6b+Xl5aqpqVFiYqICAwMVGBioDRs26Pe//70CAwPNPPhzjiSpS5cu6tmzp0tbjx49zC8U57102v/7f/9PTz75pH7xi1+oV69eys7O1uOPP64ZM2ZIIk9n405O7Ha7GhoaVFtbe84Yf+F0OjVy5Ejt3btXJSUl5io8iTyh/eI6r23hurNt4Vq47eL6vG3i88Klo4jXDgQHBysxMVElJSUu7SUlJerfv7+XZuV9hmHo0Ucf1euvv66///3v6tatm8v+bt26yW63u+StoaFBGzZs8Ju8DR48WB9++KEqKirMn759++qee+5RRUWFrr/+er/PkSQNGDBAe/bscWn7+OOPde2110rivdTs+PHj6tDB9f9WAgICzEfGk6eW3MlJYmKigoKCXGKqqqpUWVnpV3lrLuB98sknWr9+vTp37uyynzyhveI6r23gurNt4lq47eL6vG3i80IruOyP0oBHFBYWGkFBQcbixYuNXbt2Gbm5uUZYWJjx+eefe3tqXvPwww8bNpvNeO+994yqqirz5/jx42bM888/b9hsNuP11183PvzwQ+Puu+82unTpYtTX13tx5t713SdyGQY5MozTT8EMDAw0pk+fbnzyySfGihUrjNDQUGP58uVmDHkyjPvvv9+4+uqrjTVr1hh79+41Xn/9dSMqKsqYPHmyGeOPeTpy5IixY8cOY8eOHYYkY+7cucaOHTvMp6q6k5OHHnrIuOaaa4z169cb27dvN376058affr0MU6dOuWtw2p158uT0+k0MjMzjWuuucaoqKhw+W+6w+Ew+/CHPME/cZ3n+7jubD+4Fm4buD5vm/i8cOko4rUjf/jDH4xrr73WCA4ONn70ox+Zj7T3V5LO+vPqq6+aMU1NTcYzzzxj2O12w2q1Gj/5yU+MDz/80HuT9gFnXriQo9P+9re/GQkJCYbVajVuvvlmY+HChS77yZNh1NfXG4899pjRtWtXo2PHjsb1119vTJ061aXI4o95evfdd8/636L777/fMAz3cnLixAnj0UcfNSIjI42QkBAjIyPD2LdvnxeOxnPOl6e9e/ee87/p7777rtmHP+QJ/ovrPN/GdWf7wbVw28H1edvD54VLZzEMw7gcK/4AAAAAAAAAXBy+Ew8AAAAAAADwcRTxAAAAAAAAAB9HEQ8AAAAAAADwcRTxAAAAAAAAAB9HEQ8AAAAAAADwcRTxAAAAAAAAAB9HEQ8AAAAAAADwcRTxAPiF9957TxaLRYcPH76kfkaNGqU777yzVebkaZ9//rksFosqKiq8PRUAAACP4ToPgL+giAegzXnppZcUHh6uU6dOmW1Hjx5VUFCQfvzjH7vEfvDBB7JYLIqNjVVVVZVsNlurzqWmpkZjx45V165dZbVaZbfblZ6ero0bN7bqOK3lnXfeUf/+/RUeHq4uXbroiSeecMkjAACAN3Gdd3G+/fZbDR06VLGxsbJarYqLi9Ojjz6q+vp6b08NQCsK9PYEAOBCDRo0SEePHtW2bdvUr18/Sacv4ux2u7Zu3arjx48rNDRU0um/zMbGxuqmm27yyFx+9rOfyel0aunSpbr++ut18OBBvfPOOzp06JBHxrsU//znP/Xv//7vmjp1qv70pz/pyy+/1EMPPaTGxkbNnj3b29MDAADgOu8idejQQXfccYeee+45XXXVVfrXv/6lRx55RIcOHdLKlSu9PT0ArYSVeADanPj4eMXGxuq9994z29577z3dcccduuGGG1RWVubSPmjQoBa3WSxZskRXXnml3n77bfXo0UNXXHGFhg4dqqqqKvO1jY2NmjBhgq688kp17txZkydPlmEY5v7Dhw+rtLRUL7zwggYNGqRrr71Wt956q6ZMmaJhw4aZcRaLRS+++KJuv/12hYSEqFu3bvrLX/7ickxffvml7rrrLkVERKhz586644479Pnnn7vEvPrqq+rRo4c6duyom2++WX/84x9d9m/ZskW33HKLOnbsqL59+2rHjh0u+wsLC9W7d2/95je/0Y033qiUlBTNmDFDf/jDH3TkyBFJp/+Ke/fdd+uaa65RaGioevXqpT//+c8u/QwcOFDjxo1Tbm6uIiIiFBMTo4ULF+rYsWP65S9/qfDwcN1www166623vudMAgAAuOI67+Ku8yIiIvTwww+rb9++uvbaazV48GDl5OTogw8+MGOa8/LGG2/opptuUseOHZWamqr9+/ebMXl5efrhD3+oV155RV27dtUVV1yhhx9+WI2NjZo5c6bsdruio6M1ffr07zmTADyBIh6ANmngwIF69913ze13331XAwcOVEpKitne0NCgjRs3atCgQWft4/jx45o9e7aWLVum999/X/v27dOkSZPM/XPmzNErr7yixYsXq7S0VIcOHdLq1avN/VdccYWuuOIKvfHGG3I4HOed79NPP62f/exn+sc//qF7771Xd999t3bv3m3OY9CgQbriiiv0/vvvq7S01LzYbGhokCQtWrRIU6dO1fTp07V7927l5+fr6aef1tKlSyVJx44dU0ZGhuLj41VeXq68vDyXY5Ekh8Ohjh07urSFhITo5MmTKi8vlySdPHlSiYmJWrNmjSorK/Xggw8qOztbmzdvdnnd0qVLFRUVpS1btmjcuHF6+OGH9fOf/1z9+/fX9u3blZ6eruzsbB0/fvy8eQEAADgT13kXfp13pq+++kqvv/66UlJSWuRl+vTpWrp0qf73f/9X9fX1+sUvfuES8+mnn+qtt95SUVGR/vznP+uVV17RsGHDdODAAW3YsEEvvPCCfv3rX2vTpk3nnQMADzAAoA1auHChERYWZjidTqO+vt4IDAw0Dh48aBQWFhr9+/c3DMMwNmzYYEgyPv30U+Pdd981JBm1tbWGYRjGq6++akgy/vWvf5l9/uEPfzBiYmLM7S5duhjPP/+8ue10Oo1rrrnGuOOOO8y2//mf/zEiIiKMjh07Gv379zemTJli/OMf/3CZqyTjoYcecmlLSkoyHn74YcMwDGPx4sVGfHy80dTUZO53OBxGSEiI8fbbbxuGYRhxcXHGypUrXfp49tlnjeTkZMMwDOPll182IiMjjWPHjpn7X3zxRUOSsWPHDsMwDOPtt982OnToYKxcudI4deqUceDAAeO2224zJLXo+7v+/d//3Zg4caK5nZKSYtx2223m9qlTp4ywsDAjOzvbbKuqqjIkGRs3bjxnvwAAAGfDdd6FX+c1+8UvfmGEhIQYkozhw4cbJ06cMPc152XTpk1m2+7duw1JxubNmw3DMIxnnnnGCA0NNerr682Y9PR047rrrjMaGxvNtvj4eGPGjBkGgMuLlXgA2qRBgwbp2LFj2rp1qz744APddNNNio6OVkpKirZu3apjx47pvffeU9euXXX99deftY/Q0FDdcMMN5naXLl1UU1MjSaqrq1NVVZWSk5PN/YGBgerbt69LHz/72c/01Vdf6c0331R6erree+89/ehHP9KSJUtc4r7bT/N2819oy8vL9a9//Uvh4eHmX30jIyN18uRJffrpp/r666+1f/9+jR492tx/xRVX6LnnntOnn34qSdq9e7f69OljfkfM2cZMS0vTrFmz9NBDD8lqteqmm24ybwcJCAiQdPrWkunTp6t3797q3LmzrrjiChUXF2vfvn0uffXu3dv8d0BAgDp37qxevXqZbTExMZJk5hMAAMBdXOdd+HVes3nz5mn79u1644039Omnn2rChAku+888zptvvllXXnmlOV9Juu666xQeHm5ux8TEqGfPnurQoYNLG9d5wOXHgy0AtEk33nijrrnmGr377ruqra01bxWw2+3q1q2b/vd//1fvvvuufvrTn56zj6CgIJdti8Xi8l0o7mr+PpHU1FT95je/0QMPPKBnnnlGo0aNOu/rLBaLJKmpqUmJiYlasWJFi5irrrpKJ0+elHT6VoukpCSX/c3FN3fnPWHCBD3++OOqqqpSRESEPv/8c02ZMkXdunWTdPrWknnz5mn+/Pnq1auXwsLClJuba97u0exsuftu23ePDQAA4EJwnXfahV7nSadzZLfbdfPNN6tz58768Y9/rKefflpdunRpMbezzVf6/uu85jau84DLj5V4ANqs5i8yfu+99zRw4ECzPSUlRW+//bY2bdp0zu9J+T42m01dunRx+a6PU6dOmd8ddz49e/bUsWPHXNrO/M6QTZs26eabb5Yk/ehHP9Inn3yi6Oho3XjjjS4/NptNMTExuvrqq/XZZ5+12N9cfOvZs6f+8Y9/6MSJE+ccs5nFYlFsbKxCQkL05z//WXFxcfrRj34k6fTT3+644w7de++96tOnj66//np98sknbmQMAACg9XCdd3HXed/VXPz77nf6nTp1Stu2bTO39+zZo8OHD5vzBeDbKOIBaLMGDRqk0tJSVVRUuHxpb0pKihYtWqSTJ09e9MWdJD322GN6/vnntXr1an300UfKyckxn3omnX6S609/+lMtX75c//znP7V371795S9/0cyZM3XHHXe49PWXv/xFr7zyij7++GM988wz2rJlix599FFJ0j333KOoqCjdcccd+uCDD7R3715t2LBBjz32mA4cOCDp9JPCZsyYod/97nf6+OOP9eGHH+rVV1/V3LlzJUlZWVnq0KGDRo8erV27dmndunWaPXt2i2OaNWuWPvzwQ+3cuVPPPvusnn/+ef3+9783/9J74403qqSkRGVlZdq9e7fGjh2r6urqi84hAADAxeA678Ku89atW6dXX31VlZWV+vzzz7Vu3To9/PDDGjBggK677jozLigoSOPGjdPmzZu1fft2/fKXv1S/fv106623XnQuAVw+3E4LoM0aNGiQTpw4oZtvvtn8Djbp9MXdkSNHdMMNNyguLu6i+584caKqqqo0atQodejQQb/61a/0H//xH6qrq5N0+qllSUlJmjdvnj799FM5nU7FxcVpzJgxeuqpp1z6mjZtmgoLC5WTkyO73a4VK1aoZ8+ekk5/Z8v777+vJ554QiNGjNCRI0d09dVXa/DgwerUqZMk6YEHHlBoaKhmzZqlyZMnKywsTL169VJubq45l7/97W966KGHdMstt6hnz5564YUX9LOf/cxlHm+99ZamT58uh8OhPn366K9//atuv/12c//TTz+tvXv3Kj09XaGhoXrwwQd15513mscMAABwOXCdd2HXeSEhIVq0aJEef/xxORwOxcXFacSIEXryySdd5hoaGqonnnhCWVlZOnDggG677Ta98sorF51HAJeXxbiYLwYAALjNYrFo9erVuvPOO709FQAAALSitnSdt2TJEuXm5rqsOATQtnA7LQAAAAAAAODjKOIBAAAAAAAAPo7baQEAAAAAAAAfx0o8AAAAAAAAwMdRxAMAAAAAAAB8HEU8AAAAAAAAwMdRxAMAAAAAAAB8HEU8AAAAAAAAwMdRxAMAAAAAAAB8HEU8AAAAAAAAwMdRxAMAAAAAAAB8HEU8AAAAAAAAwMf9f7eSkCud10rwAAAAAElFTkSuQmCC"/>

네 특성 모두 정규 분포를 따르지 않고 왜곡된 분포이다.



따라서 IQR을 이용하여 이상치를 찾는다.



```python
# Rainfall 특성 이상치 탐색

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)

Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)

print(f"Rainfall 이상치 < {Lower_fence} or > {Upper_fence}")
```

<pre>
Rainfall 이상치 < -2.4000000000000004 or > 3.2
</pre>
**RainFall** 특성은 0부터 371 사이의 값으로 이루어져있다.



따라서 이상치의 범위는 3.2보다 큰 값이 해당된다.



```python
# Evaporation 특성 이상치 탐색

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)

Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)

print(f"Evaporation 이상치 < {Lower_fence} or > {Upper_fence}")
```

<pre>
Evaporation 이상치 < -11.800000000000002 or > 21.800000000000004
</pre>
**Evaporation** 특성은 0부터 145 사이의 값으로 이루어져있다.



따라서 이상치의 범위는 21.8보다 큰 값이 해당된다.



```python
# WindSpeed9am 특성 이상치 탐색

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)

Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)

print(f"WindSpeed9am 이상치 < {Lower_fence} or > {Upper_fence}")
```

<pre>
WindSpeed9am 이상치 < -29.0 or > 55.0
</pre>
**WindSpeed9am** 특성은 0부터 130 사이의 값으로 이루어져있다.



따라서 이상치의 범위는 55보다 큰 값이 해당된다.



```python
# WindSpeed3pm 특성 확인

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)

Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)

print(f"WindSpeed3pm 이상치 < {Lower_fence} or > {Upper_fence}")
```

<pre>
WindSpeed3pm 이상치 < -20.0 or > 57.0
</pre>
**WindSpeed3pm** 특성은 0부터 87 사이의 값으로 이루어져있다.



따라서 이상치의 범위는 57보다 큰 값이 해당된다.


> ## 입력 데이터셋과 타깃 데이터셋 선언



```python
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']
```

> ## 훈련셋과 테스트셋 분할



```python
# 훈련셋과 테스트셋의 비율을 8:2로 지정

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# 훈련셋과 테스트셋의 크기 확인

X_train.shape, X_test.shape
```

<pre>
((113754, 24), (28439, 24))
</pre>
> ## 특성 정제


특성을 정제함으로써 처음 데이터셋을 유용한 데이터셋으로 변환하여 더 잘 이해하고 예측 정확도를 높이는데 도움이 된다.



먼저 범주형과 수치형 특성을 구분한다.



```python
X_train.dtypes
```

<pre>
Location          object
MinTemp          float64
MaxTemp          float64
Rainfall         float64
Evaporation      float64
Sunshine         float64
WindGustDir       object
WindGustSpeed    float64
WindDir9am        object
WindDir3pm        object
WindSpeed9am     float64
WindSpeed3pm     float64
Humidity9am      float64
Humidity3pm      float64
Pressure9am      float64
Pressure3pm      float64
Cloud9am         float64
Cloud3pm         float64
Temp9am          float64
Temp3pm          float64
RainToday         object
Year               int64
Month              int64
Day                int64
dtype: object
</pre>

```python
# 범주형 특성

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
# 수치형 특성

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
numerical
```

<pre>
['MinTemp',
 'MaxTemp',
 'Rainfall',
 'Evaporation',
 'Sunshine',
 'WindGustSpeed',
 'WindSpeed9am',
 'WindSpeed3pm',
 'Humidity9am',
 'Humidity3pm',
 'Pressure9am',
 'Pressure3pm',
 'Cloud9am',
 'Cloud3pm',
 'Temp9am',
 'Temp3pm',
 'Year',
 'Month',
 'Day']
</pre>

```python
# 수치형 특성의 결측치 확인

X_train[numerical].isnull().sum()
```

<pre>
MinTemp            495
MaxTemp            264
Rainfall          1139
Evaporation      48718
Sunshine         54314
WindGustSpeed     7367
WindSpeed9am      1086
WindSpeed3pm      2094
Humidity9am       1449
Humidity3pm       2890
Pressure9am      11212
Pressure3pm      11186
Cloud9am         43137
Cloud3pm         45768
Temp9am            740
Temp3pm           2171
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
X_test[numerical].isnull().sum()
```

<pre>
MinTemp            142
MaxTemp             58
Rainfall           267
Evaporation      12125
Sunshine         13502
WindGustSpeed     1903
WindSpeed9am       262
WindSpeed3pm       536
Humidity9am        325
Humidity3pm        720
Pressure9am       2802
Pressure3pm       2795
Cloud9am         10520
Cloud3pm         11326
Temp9am            164
Temp3pm            555
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
# 결측치의 퍼센트 확인

for col in numerical:
    if X_train[col].isnull().mean() > 0:
        print(col, round(X_train[col].isnull().mean(),4))
```

<pre>
MinTemp 0.0044
MaxTemp 0.0023
Rainfall 0.01
Evaporation 0.4283
Sunshine 0.4775
WindGustSpeed 0.0648
WindSpeed9am 0.0095
WindSpeed3pm 0.0184
Humidity9am 0.0127
Humidity3pm 0.0254
Pressure9am 0.0986
Pressure3pm 0.0983
Cloud9am 0.3792
Cloud3pm 0.4023
Temp9am 0.0065
Temp3pm 0.0191
</pre>
데이터가 무작위로 누락되었다는 가정하에 결측치를 추정값으로 대체하는 방법에는 두 가지가 존재한다.



하나는 평균값 또는 중앙값으로 대체하는 것이고, 다른 하나는 무작위 샘플 추정이다.



데이터셋에 이상치가 존재하는 경우 중앙값으로 대체하는 것이 좋다.



따라서 중앙값으로 결측치를 대체한다.



과적합을 피하기 위해서는 추정값으로 결측치를 대체할 때 훈련셋은 훈련셋에 대하여, 테스트셋은 테스트셋에 대하여 추정해야한다.






```python
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)
```


```python
# 결측치가 존재하는지 다시한번 확인

X_train[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
X_test[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
# 범주형 특성의 결측치 확인

X_train[categorical].isnull().sum()
```

<pre>
Location          0
WindGustDir    7407
WindDir9am     7978
WindDir3pm     3008
RainToday      1139
dtype: int64
</pre>

```python
# 결측치의 퍼센트 확인

for col in categorical:
    if X_train[col].isnull().mean() > 0:
        print(col, (X_train[col].isnull().mean()))
```

<pre>
WindGustDir 0.06511419378659213
WindDir9am 0.07013379749283542
WindDir3pm 0.026443026179299188
RainToday 0.01001283471350458
</pre>

```python
# 결측치를 빈도수가 가장 높은 값으로 대체

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```


```python
# 결측치가 있는지 다시한번 확인

X_train[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>

```python
X_test[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>

```python
# 전체 데이터셋의 결측치 확인

X_train.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
X_test.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>
수치형 특성의 이상치 문제 해결



앞서 **Rainfall**, **Evaporation**, **WindSpeed9am**, **WindSpeed3pm** 각각의 특성에 이상치가 존재하는 것을 알았다.



top-coding 방식을 이용하여 최댓값을 제한하고 이상치를 제거한다.



```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```

<pre>
(3.2, 3.2)
</pre>

```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```

<pre>
(21.8, 21.8)
</pre>

```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```

<pre>
(55.0, 55.0)
</pre>

```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```

<pre>
(57.0, 57.0)
</pre>

```python
X_train[numerical].describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>1017.640649</td>
      <td>1015.241101</td>
      <td>4.651801</td>
      <td>4.703588</td>
      <td>16.995062</td>
      <td>21.688643</td>
      <td>2012.759727</td>
      <td>6.404021</td>
      <td>15.710419</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>6.738680</td>
      <td>6.675168</td>
      <td>2.292726</td>
      <td>2.117847</td>
      <td>6.463772</td>
      <td>6.855649</td>
      <td>2.540419</td>
      <td>3.427798</td>
      <td>8.796821</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>980.500000</td>
      <td>977.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.200000</td>
      <td>-5.400000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>1013.500000</td>
      <td>1011.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>12.300000</td>
      <td>16.700000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>1017.600000</td>
      <td>1015.200000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>16.700000</td>
      <td>21.100000</td>
      <td>2013.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>1021.800000</td>
      <td>1019.400000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>21.500000</td>
      <td>26.300000</td>
      <td>2015.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1041.000000</td>
      <td>1039.600000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>40.200000</td>
      <td>46.700000</td>
      <td>2017.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
    </tr>
  </tbody>
</table>
</div>


이상치가 존재하였던 특성들이 상한선을 넘은 것을 볼 수 있다.



```python
# 범주형 특성 인코딩

categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
X_train[categorical].head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>S</td>
      <td>SSE</td>
      <td>S</td>
      <td>No</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>ENE</td>
      <td>SSE</td>
      <td>SE</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>E</td>
      <td>NE</td>
      <td>N</td>
      <td>No</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>ESE</td>
      <td>SSE</td>
      <td>E</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>W</td>
      <td>N</td>
      <td>SE</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



```python
# RainToday 특성 인코딩

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)
X_test = encoder. fit_transform(X_test)
```


```python
X_train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday_0</th>
      <th>RainToday_1</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>Witchcliffe</td>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>S</td>
      <td>41.0</td>
      <td>SSE</td>
      <td>S</td>
      <td>...</td>
      <td>1013.4</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>18.8</td>
      <td>20.4</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>4</td>
      <td>25</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>Cairns</td>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>ENE</td>
      <td>33.0</td>
      <td>SSE</td>
      <td>SE</td>
      <td>...</td>
      <td>1013.1</td>
      <td>7.0</td>
      <td>5.0</td>
      <td>26.4</td>
      <td>27.5</td>
      <td>1</td>
      <td>0</td>
      <td>2015</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>AliceSprings</td>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>E</td>
      <td>31.0</td>
      <td>NE</td>
      <td>N</td>
      <td>...</td>
      <td>1013.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.5</td>
      <td>35.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>Cairns</td>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>ESE</td>
      <td>37.0</td>
      <td>SSE</td>
      <td>E</td>
      <td>...</td>
      <td>1010.8</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>27.3</td>
      <td>29.4</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>Newcastle</td>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>W</td>
      <td>39.0</td>
      <td>N</td>
      <td>SE</td>
      <td>...</td>
      <td>1015.2</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>22.2</td>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>11</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>


 **RainToday** 특성이 **RainToday_0** 과 **RainToday_1**으로 대체된 것을 확인할 수 있다.


나머지 범주형 특성들은 원-핫 인코딩을 이용하고 훈련셋을 구성한다.



```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                   pd.get_dummies(X_train.Location),
                   pd.get_dummies(X_train.WindGustDir),
                   pd.get_dummies(X_train.WindDir9am),
                   pd.get_dummies(X_train.WindDir3pm)], axis=1)
```

테스트셋 또한 똑같은 과정을 적용한다.



```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```


```python
X_train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110803</th>
      <td>13.9</td>
      <td>22.6</td>
      <td>0.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>28.0</td>
      <td>65.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87289</th>
      <td>22.4</td>
      <td>29.4</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.3</td>
      <td>33.0</td>
      <td>7.0</td>
      <td>19.0</td>
      <td>71.0</td>
      <td>59.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>134949</th>
      <td>9.7</td>
      <td>36.2</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>12.3</td>
      <td>31.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>85553</th>
      <td>20.5</td>
      <td>30.1</td>
      <td>0.0</td>
      <td>8.8</td>
      <td>11.1</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>19.0</td>
      <td>59.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16110</th>
      <td>16.8</td>
      <td>29.2</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>72.0</td>
      <td>53.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>



```python
X_test.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86232</th>
      <td>17.4</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>3.6</td>
      <td>11.1</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>63.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57576</th>
      <td>6.8</td>
      <td>14.4</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>8.5</td>
      <td>46.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>80.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124071</th>
      <td>10.1</td>
      <td>15.4</td>
      <td>3.2</td>
      <td>4.8</td>
      <td>8.5</td>
      <td>31.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>70.0</td>
      <td>61.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117955</th>
      <td>14.4</td>
      <td>33.4</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>11.6</td>
      <td>41.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>40.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>133468</th>
      <td>6.8</td>
      <td>14.3</td>
      <td>3.2</td>
      <td>0.2</td>
      <td>7.3</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>92.0</td>
      <td>47.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>


> ## 특성 스케일링



```python
X_train.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.193497</td>
      <td>23.237216</td>
      <td>0.675080</td>
      <td>5.151606</td>
      <td>8.041154</td>
      <td>39.884074</td>
      <td>13.978155</td>
      <td>18.614756</td>
      <td>68.867486</td>
      <td>51.509547</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.388279</td>
      <td>7.094149</td>
      <td>1.183837</td>
      <td>2.823707</td>
      <td>2.769480</td>
      <td>13.116959</td>
      <td>8.806558</td>
      <td>8.685862</td>
      <td>18.935587</td>
      <td>20.530723</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.200000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.600000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.800000</td>
      <td>8.500000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.400000</td>
      <td>8.700000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>33.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>



```python
cols_train = X_train.columns
cols_test = X_test.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
```


```python
X_train = pd.DataFrame(X_train, columns=[cols_train])
```


```python
X_test = pd.DataFrame(X_test, columns=[cols_test])
```


```python
X_train.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>...</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
      <td>113754.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.484406</td>
      <td>0.530004</td>
      <td>0.210962</td>
      <td>0.236312</td>
      <td>0.554562</td>
      <td>0.262667</td>
      <td>0.254148</td>
      <td>0.326575</td>
      <td>0.688675</td>
      <td>0.515095</td>
      <td>...</td>
      <td>0.054530</td>
      <td>0.060288</td>
      <td>0.067259</td>
      <td>0.101605</td>
      <td>0.064059</td>
      <td>0.056402</td>
      <td>0.064464</td>
      <td>0.069334</td>
      <td>0.060798</td>
      <td>0.065483</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.151741</td>
      <td>0.134105</td>
      <td>0.369949</td>
      <td>0.129528</td>
      <td>0.190999</td>
      <td>0.101682</td>
      <td>0.160119</td>
      <td>0.152384</td>
      <td>0.189356</td>
      <td>0.205307</td>
      <td>...</td>
      <td>0.227061</td>
      <td>0.238021</td>
      <td>0.250471</td>
      <td>0.302130</td>
      <td>0.244860</td>
      <td>0.230698</td>
      <td>0.245578</td>
      <td>0.254022</td>
      <td>0.238960</td>
      <td>0.247378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.375297</td>
      <td>0.431002</td>
      <td>0.000000</td>
      <td>0.183486</td>
      <td>0.565517</td>
      <td>0.193798</td>
      <td>0.127273</td>
      <td>0.228070</td>
      <td>0.570000</td>
      <td>0.370000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.479810</td>
      <td>0.517958</td>
      <td>0.000000</td>
      <td>0.220183</td>
      <td>0.586207</td>
      <td>0.255814</td>
      <td>0.236364</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.520000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.593824</td>
      <td>0.623819</td>
      <td>0.187500</td>
      <td>0.247706</td>
      <td>0.600000</td>
      <td>0.310078</td>
      <td>0.345455</td>
      <td>0.421053</td>
      <td>0.830000</td>
      <td>0.650000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>


> ## 모델 훈련



```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear', random_state=0)

logreg.fit(X_train, y_train)
```

<pre>
LogisticRegression(random_state=0, solver='liblinear')
</pre>
> ## 모델 예측



```python
y_pred_test = logreg.predict(X_test)
y_pred_test
```

<pre>
array(['No', 'No', 'No', ..., 'No', 'No', 'Yes'], dtype=object)
</pre>
predict_proba 메소드



해당 메소드는 목표 변수(타깃 데이터)에 대한 확률을 배열 형태로 보여준다.



```python
logreg.predict_proba(X_test)[:,0]
```

<pre>
array([0.92190214, 0.85735432, 0.8511671 , ..., 0.98224021, 0.82899741,
       0.34815758])
</pre>

```python
logreg.predict_proba(X_test)[:,1]
```

<pre>
array([0.07809786, 0.14264568, 0.1488329 , ..., 0.01775979, 0.17100259,
       0.65184242])
</pre>
> ## 정확도 측정



```python
from sklearn.metrics import accuracy_score

print(f"모델 정확도 : {accuracy_score(y_test, y_pred_test):0.4f}")
```

<pre>
모델 정확도 : 0.8488
</pre>
y_test는 실제 데이터, y_pred_test는 예측한 데이터이다.


훈련셋과 테스트셋의 정확도를 비교하여 과적합 여부를 판단한다.



```python
y_pred_train = logreg.predict(X_train)
y_pred_train
```

<pre>
array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)
</pre>

```python
print(f"훈련 셋 정확도 : {accuracy_score(y_train, y_pred_train):0.4f}")
```

<pre>
훈련 셋 정확도 : 0.8477
</pre>
과대적합 및 과소적합 여부 확인



```python
print(f"훈련 셋 점수 : {logreg.score(X_train, y_train):0.4f}")
print(f"테스트 셋 점수 : {logreg.score(X_test, y_test):0.4f}")
```

<pre>
훈련 셋 점수 : 0.8477
테스트 셋 점수 : 0.8488
</pre>
훈련 셋과 테스트 셋의 점수 차이가 약 0.01밖에 차이가 나지 않으므로 과대적합은 아니다.



로지스틱 회귀에서 C는 기본값을 1로 사용한다.



훈련 셋 및 테스트 셋의 정확도는 약 85%로 우수한 성능이지만 두 셋의 정확도 차이가 얼마 나지 않으므로 과소적합을 의심해 볼 수 있다.



C 값에 변화를 주면서 살펴본다.



```python
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)

logreg100.fit(X_train, y_train)
```

<pre>
LogisticRegression(C=100, random_state=0, solver='liblinear')
</pre>

```python
print(f"훈련 셋 점수 : {logreg100.score(X_train, y_train):0.4f}")
print(f"테스트 셋 점수 : {logreg100.score(X_test, y_test):0.4f}")
```

<pre>
훈련 셋 점수 : 0.8478
테스트 셋 점수 : 0.8478
</pre>

```python
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)

logreg001.fit(X_train, y_train)
```

<pre>
LogisticRegression(C=0.01, random_state=0, solver='liblinear')
</pre>

```python
print(f"훈련 셋 점수 : {logreg001.score(X_train, y_train):0.4f}")
print(f"테스트 셋 점수 : {logreg001.score(X_test, y_test):0.4f}")
```

<pre>
훈련 셋 점수 : 0.8409
테스트 셋 점수 : 0.8441
</pre>
C 값을 증가시켜 100으로 지정하였을 때 훈련 셋 점수는 아주 조금 올라갔고, 테스트 셋 점수는 내려갔다.



0.01로 지정하였을 때는 두 훈련 셋 모두 점수가 내려간 결과를 볼 수 있다.


모델 정확도와 null 정확도 비교



현재 모델 정확도는 0.8488로, 모델 정확도만을 가지고 매우 우수하다고 볼 수 없다.



null 정확도와 비교해야 하는데, 가장 빈번한 클래스를 예측함으로써 얻을 수 있다.



```python
y_test.value_counts()
```

<pre>
No     22067
Yes     6372
Name: RainTomorrow, dtype: int64
</pre>
클래스에서 가장 빈번하게 발생하는 횟수가 22067번임을 알 수 있다.



이를 총 발생횟수로 나누면 null 정확도를 계산할 수 있다.



```python
null_accuracy = (22067/(22067+7372))

print(f"Null 정확도 : {null_accuracy:0.4f}")
```

<pre>
Null 정확도 : 0.7496
</pre>
모델 정확도는 0.8488, null 정확도는 0.7496으로 로지스틱 회귀 모델이 타깃을 예측하는데 잘 작동되고 있음을 알 수 있다.



아를 통해 분류 모델의 정확도가 좋음을 알 수 있고, 타깃을 예측하는 면에서도 잘 작동한다.



하지만, 기본 값의 분포는 주지않고, 또한 분류기가 어떤 유형의 오류를 범하는지 알려주지 않는다.


> ## 혼동 행렬(Confusion Matrix)


혼동 행렬을 이용하여 분류 모델의 성능과 모델에서 발생하는 오류 유형을 파악할 수 있다.



각 특성별로 분류된 올바른 예측과 잘못된 예측에 대한 요약을 표 형식으로 제공해준다.



분류 모델의 성능을 평가하는 동안 4가지 유형의 결과가 나올 수 있는데, 다음과 같다.



1. TP

관찰이 특정 클래스에 속할 것으로 예측하고 실제로 해당 클래스에 속할 때 발생



2. TN

관찰이 특정 글래스에 속하지 않는다고 예측하고 실제로 해당 클래스에 속하지 않을 때 발생



3. FP

관찰이 특정 클래스에 속할 것으로 예측하였으나 실제로 해당 클래스에 속하지 않을 때 발생



4. FN

관찰이 특정 클래스에 속하지 않는다고 예측하였으나 실제로 해당 클래스에 속할 때 발생



이 중 FN의 경우 심각한 오류이며 유형 Ⅱ오류라고 한다.



```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

<pre>
Confusion matrix

 [[21109   958]
 [ 3343  3029]]

True Positives(TP) =  21109

True Negatives(TN) =  3029

False Positives(FP) =  958

False Negatives(FN) =  3343
</pre>
24177개의 올바른 예측과 4262개의 잘못된 예측을 보여준다.



```python
# heatmap으로 시각화

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

<pre>
<AxesSubplot:>
</pre>
<img src="data:image/png;base64,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"/>

> ## 분류 행렬


분류 보고서는 분류 모델의 성능을 평가하는 방법 중 하나이다.



모델의 정밀도, 재검색, F1 및 지원 점수가 표시된다.



다음의 코드로 표현할 수 있다.



```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

<pre>
              precision    recall  f1-score   support

          No       0.86      0.96      0.91     22067
         Yes       0.76      0.48      0.58      6372

    accuracy                           0.85     28439
   macro avg       0.81      0.72      0.75     28439
weighted avg       0.84      0.85      0.84     28439

</pre>
분류 정확도



```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('분류 정확도 : {0:0.4f}'.format(classification_accuracy))
```

<pre>
분류 정확도 : 0.8488
</pre>
분류 오류



```python
classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('분류 오류 : {0:0.4f}'.format(classification_error))
```

<pre>
분류 오류 : 0.1512
</pre>
정확도



정확도는 예측된 모든 양성(P) 결과 중 올바르게 예측된 양성 결과의 비율로 정의한다.



즉, TP + FP에 대한 TP의 비율이다.



이는 정확하게 예측된 긍정적인 결과의 비율을 식별하는 것으로, N 보다는 P에 집중된다.



```python
precision = TP / float(TP + FP)


print('정확도 : {0:0.4f}'.format(precision))
```

<pre>
정확도 : 0.9566
</pre>
리콜(민감도)



리콜은 실제 양성 결과 중 올바르게 예측된 양성 결과의 비율로 정의된다.



즉, TP + FN에 대한 TP의 비율이다.



```python
recall = TP / float(TP + FN)

print('리콜(민감도) : {0:0.4f}'.format(recall))
```

<pre>
리콜(민감도) : 0.8633
</pre>
TP 비율



TP 비율과 리콜(민감도)는 동의어이다.



```python
true_positive_rate = TP / float(TP + FN)


print('TP 비율 : {0:0.4f}'.format(true_positive_rate))
```

<pre>
TP 비율 : 0.8633
</pre>
FP 비율



```python
false_positive_rate = FP / float(FP + TN)


print('FP 비율 : {0:0.4f}'.format(false_positive_rate))
```

<pre>
FP 비율 : 0.2403
</pre>
특이점



```python
specificity = TN / (TN + FP)

print('특이점 : {0:0.4f}'.format(specificity))
```

<pre>
특이점 : 0.7597
</pre>
f1-score



f1-score는 정확도와 리콜의 조화평균으로, 최댓값 1, 최저값 0을 가진다.



따라서 f1-score는 정확도와 리콜을 계산에 포함하기에 정확도 측정값보다 항상 낮은 값을 갖는다.



분류기 모델을 비교할 때에는 f1-score의 가중 평균을 사용하여 비교해야한다.


Support



Support는 데이터 셋에서 클래스의 실제 발생 횟수를 의미한다.


> ## 임계값 레벨 조정



```python
# 처음 10개의 예측된 확률값 출력

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

<pre>
array([[0.92190214, 0.07809786],
       [0.85735432, 0.14264568],
       [0.8511671 , 0.1488329 ],
       [0.99167847, 0.00832153],
       [0.9649014 , 0.0350986 ],
       [0.98227531, 0.01772469],
       [0.2080994 , 0.7919006 ],
       [0.26033723, 0.73966277],
       [0.91874581, 0.08125419],
       [0.8849659 , 0.1150341 ]])
</pre>
* 각 행 별로 합은 1이다.



* 0과 1을 특성으로 가지는 2개의 열이 존재한다.



    * 0 : 내일 비가 내리지 않을 것으로 예상되는 확률



    * 1 : 내일 비가 올 것으로 예상되는 확률



* 예측된 확률의 중요성



    * 비가 올 확률 or 오지 않을 확률에 따라 순위를 매길 수 있다.

    

* predict_proba 프로세스



    * 확률을 예측

    

    * 가장 높은 확률을 가진 클래스 선택

    

* 분류 임계값 레벨



    * 분류 임계값의 기본값은 0.5이다.

    

    * 1 : 확률이 0.5를 초과하면 비가 온다고 예측

    

    * 0 : 확률이 0.5 미만이면 비가 내리지 않는다고 예측



```python
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prob of - No rain tomorrow (0)</th>
      <th>Prob of - Rain tomorrow (1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.921902</td>
      <td>0.078098</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.857354</td>
      <td>0.142646</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.851167</td>
      <td>0.148833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.991678</td>
      <td>0.008322</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.964901</td>
      <td>0.035099</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.982275</td>
      <td>0.017725</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.208099</td>
      <td>0.791901</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.260337</td>
      <td>0.739663</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.918746</td>
      <td>0.081254</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.884966</td>
      <td>0.115034</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 비가 온다고 예측한 것에  처음 10개의 예측 확률

logreg.predict_proba(X_test)[0:10, 1]
```

<pre>
array([0.07809786, 0.14264568, 0.1488329 , 0.00832153, 0.0350986 ,
       0.01772469, 0.7919006 , 0.73966277, 0.08125419, 0.1150341 ])
</pre>

```python
# 저장

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```

히스토그램으로 보면 다음과 같다.



```python
plt.rcParams['font.size'] = 12

plt.hist(y_pred1, bins = 10)

plt.title('Histogram of predicted probabilities of rain')

plt.xlim(0,1)

plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

<pre>
Text(0, 0.5, 'Frequency')
</pre>
<img src="data:image/png;base64,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vlytPC8vT/Tr108oFApx/PhxqVz1fu7atavIy8uTytPT08Uzzzyjtb/lUf3uABBvv/222rKkpCRhYGAgrK2tRVZWllT+xRdfCABCoVCIHTt2aLRZ2fdT2d+d1atXq62zbds2AUA0b95clJSUSOW3b98W2dnZGtv+448/hKmpqejXr5/O9vOLL75Qq6/tvV+ZumVpO1eq98u0adNEcXGxVF5cXCzGjx8vAIi4uDip/NlnnxVGRkbi9u3bGu3fuXOn3G2XtXDhQgFADBo0SO337NatW6Jp06YCgDhw4IDaOqrf58pQvdednJw0PlOEePC7c/36dY3yjIwM0bp1a2FjYyNyc3PVlj3qfOjr64uEhAS1ZeHh4QKAWLJkSaX6XhsxkNVjqg+8irweF8j++usvAUC8/PLLj93u4wKZn5+fACB2796tseyXX34RAISvr69UNn/+fAFAREVFadRPTk4W+vr65Qaydu3aPba/D0tLSxMAREhIiFq56oN59uzZGutERUVJQeVhX375pQAgYmNjH7vtgoICYWJiIszNzdVCksp7772n9VhUJ5A9HLqE+Pf4+fj4qJUDKPfLZvr06QKA+Omnn7Rub+jQoUJPT0/6kv3tt98EANGrVy+NusXFxcLNza1Cgay4uFhYWVkJExMTkZKS8rjdfmQgS0tLE/r6+qJTp05a1z158qRGiPD39xcAxN69ezXqq76IKhvIrKystAYc1Rdo2feSahva/rFUlfeT6tw/HLpUVO+bsv/weJRBgwYJpVIpCgsLdbKfTzKQlZSUCDs7O9GoUSO1MKZy9+5doVAoxAsvvCCVdejQQZiammo93hXl5uYmFAqFOH/+vMaytWvXav18qk4g++ijjyrdx+XLlwsAYt++fWrljzofY8aM0Wjn8uXLAoAYPnx4pftQ2/CWJUEIUe4yFxcXXL169bFttGnTBs8++yw2bdqE69evY8iQIejevTuee+65Sg8kP3HiBPT09NC7d2+NZb6+vtDX18fx48fV6gNAjx49NOo7OzujadOm5U7b0aVLl3L7cf/+faxcuRJxcXH4559/kJOTo3asUlJStK7XsWNHjTLVQNpHLfvf//5Xbl9Uzp8/j7y8PPTo0QM2NjYay/39/bFo0SK141Nd2s5Dz549YWBgIB37slxcXKRblGUdOnQIwIPxOkePHtVYnpqaitLSUly4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"/>

* 위의 히스토그램에서 P쪽으로 매우 치우쳐 있다.



* 첫 번째 열은 확률이 0.0 ~ 0.1 사이의 값이 약 15000개임을 알려준다.



* 확률이 0.5를 넘는 값도 존재한다.



* 0.5를 넘는 값은 내일 비가 온다고 예측한다.



* 대부분의 값은 내일 비가 오지 않는다고 예측한다.



```python
import sklearn.preprocessing

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = sklearn.preprocessing.binarize(y_pred1, threshold=i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('임계값이 ',i/10,' 일 때 분류 행렬 ','\n\n',cm1,'\n\n',
           
            '올바른 예측의 개수 : ',cm1[0,0]+cm1[1,1], '\n\n', 
           
            '유형 1 오류( False Positives), ', cm1[0,1],'\n\n',
           
            '유형 2 오류( False Negatives), ', cm1[1,0],'\n\n',
           
           '정확도 : ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           '민감도: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           '특이점: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

<pre>
임계값이  0.1  일 때 분류 행렬  

 [[13876  8191]
 [  678  5694]] 

 올바른 예측의 개수 :  19570 

 유형 1 오류( False Positives),  8191 

 유형 2 오류( False Negatives),  678 

 정확도 :  0.6881395267062836 

 민감도:  0.8935969868173258 

 특이점:  0.6288122535913355 

 ==================================================== 


임계값이  0.2  일 때 분류 행렬  

 [[17849  4218]
 [ 1458  4914]] 

 올바른 예측의 개수 :  22763 

 유형 1 오류( False Positives),  4218 

 유형 2 오류( False Negatives),  1458 

 정확도 :  0.8004149231688878 

 민감도:  0.7711864406779662 

 특이점:  0.8088548511351792 

 ==================================================== 


임계값이  0.3  일 때 분류 행렬  

 [[19590  2477]
 [ 2161  4211]] 

 올바른 예측의 개수 :  23801 

 유형 1 오류( False Positives),  2477 

 유형 2 오류( False Negatives),  2161 

 정확도 :  0.8369140968388481 

 민감도:  0.6608600125549278 

 특이점:  0.8877509403181221 

 ==================================================== 


임계값이  0.4  일 때 분류 행렬  

 [[20506  1561]
 [ 2752  3620]] 

 올바른 예측의 개수 :  24126 

 유형 1 오류( False Positives),  1561 

 유형 2 오류( False Negatives),  2752 

 정확도 :  0.8483420654734696 

 민감도:  0.5681104833647207 

 특이점:  0.929260887297775 

 ==================================================== 


</pre>
* 이진 문제에서는 기본적으로 0.5의 임계값이 사용되어 예측 확률을 클래스 확률로 변환한다.



* 임계값을 조정하여 민감도 or 특이점을 높일 수 있다.



* 민감도와 특이점은 반비례 관계이다.



* 임계값 수준을 높이면 정확도가 높아진다.



* 임계값 레벨을 조정하는 것은 모델 구축의 마지막 단계에서 수행해야한다.


> ## ROC 


분류 모델의 성능을 시각적으로 측정하는 다른 방법은 ROC 곡선이다.



ROC 곡선은 다양한 임계값 레벨에서 분류 모델의 성능을 보여주는 도표이고, 이는 FPR에 대한 TPR을 배치한다.



또한 단일 포인트의 TPR과 FPR에 초점을 맞춰, 다양한 임계값에 대한 일반적인 성능을 알 수 있고 이를 그래프로 표현한다.



여기서 임계값을 낮추면 더 많은 항목이 P로 분류되어 TP와 FP가 증가한다.



따라서 특정 부분에 대한 민감도와 특이점의 균형을 맞추는 임계값을 지정하는데 도움을 줄 수 있다.



```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
```

<img src="data:image/png;base64,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"/>

ROC의 AUC



ROC의 AUC는 ROC 곡선 아래의 면적을 의미하는데 분류기 성능을 비교할 수 있다.



완벽한 분류기는 AUC가 1이고, 순수 무작위 분류기는 0.5이다.



```python
from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

<pre>
ROC AUC : 0.8730
</pre>
ROC의 AUC 값이 높을수록 더 좋은 분류기이다.



현재 0.8730으로 괜찮은 성능을 내고 있다.



밑의 코드는 cross_val_score 메소드를 이용하여 계산한다.



```python
from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

<pre>
Cross validated ROC AUC : 0.8695
</pre>
> ## k-겹 교차검증



```python
# 폴드 5개를 적용하여 교차검증

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation 점수 : {}'.format(scores))
```

<pre>
Cross-validation 점수 : [0.84686387 0.84624852 0.84633642 0.84963298 0.84773626]
</pre>
교차 검증의 정확도는 평균을 계산하여 정리가능하다.



```python
print('cross-validation 평균 점수: {:.4f}'.format(scores.mean()))
```

<pre>
cross-validation 평균 점수: 0.8474
</pre>
> ##  그리드탐색으로 하이퍼파라미터 미세조정



```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```

<pre>
GridSearchCV(cv=5,
             estimator=LogisticRegression(random_state=0, solver='liblinear'),
             param_grid=[{'penalty': ['l1', 'l2']}, {'C': [1, 10, 100, 1000]}],
             scoring='accuracy')
</pre>

```python
print(f"GridSearch CV best score : {grid_search.best_score_:.4f}\n\n")

print("Best score에 대한 Parameters : ", grid_search.best_params_)

print("\n\n탐색에 의해 선택된 Estimator : ", grid_search.best_estimator_)
```

<pre>
GridSearch CV best score : 0.8474


Best score에 대한 Parameters :  {'penalty': 'l2'}


탐색에 의해 선택된 Estimator :  LogisticRegression(random_state=0, solver='liblinear')
</pre>

```python
print(f"테스트 셋에 대한 GridSearch CV 점수 : {grid_search.score(X_test, y_test):.4f}")
```

<pre>
테스트 셋에 대한 GridSearch CV 점수 : 0.8488
</pre>
* 처음 모델 테스트 정확도와 그리드 탐색을 통한 정확도는 0.8488로 동일하다.



* 특정 모델에서는 그리드 탐색을 수행할 경우 성능을 향상시킬 수 있다.


> ## 결과/결론


* 로지스틱 회귀 모델의 정확도는 0.8488로 내일 호주에 비가 올지 예측하는데 잘 작동한다.



* 일부 관측값은 내일 비가 올 것이라고 예측하고, 대부분의 관측값은 비가 오지 않을 것이라고 예측한다.



* 이번 데이터에 대한 모델은 과대적합을 보이지 않는다.



* C 값을 적당한 값으로 조정하여 테스트 셋의 정확도를 높이도록 설계할 수 있다.



* 임계값을 높이면 정확도가 높아진다.



* ROC 곡선의 AUC가 1에 가까우므로 내일 비가 올지 안올지 예측하는데에 잘 작동된다는 것을 알 수 있다.



* 원래 모델의 점수는 0.8488이고 평균 교차 검증을 수행하였을 경우 0.8474로 교차 검증이 성능 향상으로 이어지지는 않는다.



* 원래 모델 테스트 정확도는 0.8488이고, 그리드 탐색을 통한 정확도는 0.8488로 동일한 결과값을 얻었다.

