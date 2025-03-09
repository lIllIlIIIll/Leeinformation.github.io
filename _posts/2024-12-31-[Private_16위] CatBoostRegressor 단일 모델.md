---
layout: single
title:  "[데이콘-Private 16위] 웹 로그 기반 조회수 예측"
categories: AI
tag: [python, Machine Learning]
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


# 1. 데이터 다운로드 및 라이브러리 불러오기



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
```


```python
df_train = pd.read_csv(os.path.join("train.csv"))
df_test = pd.read_csv(os.path.join("test.csv"))
```


```python
df_train.shape
```

<pre>
(252289, 19)
</pre>

```python
df_train.head(5)
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
      <th>sessionID</th>
      <th>userID</th>
      <th>TARGET</th>
      <th>browser</th>
      <th>OS</th>
      <th>device</th>
      <th>new</th>
      <th>quality</th>
      <th>duration</th>
      <th>bounced</th>
      <th>transaction</th>
      <th>transaction_revenue</th>
      <th>continent</th>
      <th>subcontinent</th>
      <th>country</th>
      <th>traffic_source</th>
      <th>traffic_medium</th>
      <th>keyword</th>
      <th>referral_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SESSION_000000</td>
      <td>USER_000000</td>
      <td>17.0</td>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>0</td>
      <td>45.0</td>
      <td>839.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>google</td>
      <td>organic</td>
      <td>Category8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SESSION_000001</td>
      <td>USER_000001</td>
      <td>3.0</td>
      <td>Chrome</td>
      <td>Windows</td>
      <td>desktop</td>
      <td>1</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Europe</td>
      <td>Western Europe</td>
      <td>Germany</td>
      <td>google</td>
      <td>organic</td>
      <td>Category8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SESSION_000002</td>
      <td>USER_000002</td>
      <td>1.0</td>
      <td>Samsung Internet</td>
      <td>Android</td>
      <td>mobile</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Asia</td>
      <td>Southeast Asia</td>
      <td>Malaysia</td>
      <td>(direct)</td>
      <td>(none)</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SESSION_000003</td>
      <td>USER_000003</td>
      <td>1.0</td>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>Partners</td>
      <td>affiliate</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SESSION_000004</td>
      <td>USER_000004</td>
      <td>1.0</td>
      <td>Chrome</td>
      <td>iOS</td>
      <td>mobile</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>groups.google.com</td>
      <td>referral</td>
      <td>NaN</td>
      <td>Category6_Path_0000</td>
    </tr>
  </tbody>
</table>
</div>


# 2. 전처리



```python
from sklearn.preprocessing import StandardScaler

categorical_features = [
    "browser",
    "OS",
    "device",
    "subcontinent",
    "country",
    "traffic_source",
    "traffic_medium",
    "referral_path",
    "keyword",
    "continent"
]

numerical_features = [
    "quality",
    "transaction_revenue"
]

def drop_column(data) :
    data = data.drop(["sessionID", "userID"], axis=1)
    return data

def select_target(data) :
    target = data["TARGET"]
    data = data.drop(["TARGET"], axis=1)
    return data, target

def filled_nan(data) :
    data = data.fillna("N")
    return data

def not_set_preprocessing(data):
    for column in data.columns :
        data = data[data[column] != "(not set)"]
    return data

def change_type(data) :
    for i in categorical_features :
        data[i] = data[i].astype("category")
    return data

def new_seperate(data) :
    data.loc[data["subcontinent"].isin(["South America", "Central America", "Caribbean"]), "subcontinent"] = "Latin America"
    data.loc[data["subcontinent"].isin(["Southern Africa", "Western Africa", "Eastern Africa", "Middle Africa"]), "subcontinent"] = "Sub-Saharan Africa"
    return data

def numerical_preprocessing(data) :
    data["duration_quality_ratio"] = data["quality"] / data["duration"]
    data.loc[data["duration_quality_ratio"] == np.inf, "duration_quality_ratio"] = 1
    
    # 거래 수익은 0이지만 거래 횟수가 1 이상 전처리
    condition = (data["transaction"] >= 1.0) & (data["transaction_revenue"] == 0)
    data.loc[condition, "transaction"] = 0.0
    return data
    
def standard_scaler(train_data, test_data) :
    scaler = StandardScaler()

    train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
    test_data[numerical_features] = scaler.transform(test_data[numerical_features])
    
    return train_data, test_data
```


```python
# quality와 duration 값에 비해 타겟이 너무 높은 이상치 제거
df_train = df_train[df_train["TARGET"] != 386.0]
df_train = df_train[df_train["TARGET"] != 283.0]
```


```python
# 훈련셋 전처리
df_train = drop_column(df_train)
df_train = filled_nan(df_train)
df_train = not_set_preprocessing(df_train)
df_train = new_seperate(df_train)
df_train = change_type(df_train)
df_train = numerical_preprocessing(df_train)

# 테스트셋 전처리
df_test = drop_column(df_test)
df_test = filled_nan(df_test)
df_test = new_seperate(df_test)
df_test = change_type(df_test)
df_test = numerical_preprocessing(df_test)
```


```python
# 타깃 설정
train_target = select_target(df_train)
df_train = train_target[0]
target = train_target[1]
```


```python
# 수치형 특성 전처리
scaled_data = standard_scaler(df_train, df_test)
df_train = scaled_data[0]
df_test = scaled_data[1]
```


```python
df_train
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
      <th>browser</th>
      <th>OS</th>
      <th>device</th>
      <th>new</th>
      <th>quality</th>
      <th>duration</th>
      <th>bounced</th>
      <th>transaction</th>
      <th>transaction_revenue</th>
      <th>continent</th>
      <th>subcontinent</th>
      <th>country</th>
      <th>traffic_source</th>
      <th>traffic_medium</th>
      <th>referral_path</th>
      <th>duration_quality_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>0</td>
      <td>3.677386</td>
      <td>839.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>google</td>
      <td>organic</td>
      <td>N</td>
      <td>0.053635</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chrome</td>
      <td>Windows</td>
      <td>desktop</td>
      <td>1</td>
      <td>-0.255574</td>
      <td>39.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Europe</td>
      <td>Western Europe</td>
      <td>Germany</td>
      <td>google</td>
      <td>organic</td>
      <td>N</td>
      <td>0.025641</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Samsung Internet</td>
      <td>Android</td>
      <td>mobile</td>
      <td>1</td>
      <td>-0.255574</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Asia</td>
      <td>Southeast Asia</td>
      <td>Malaysia</td>
      <td>(direct)</td>
      <td>(none)</td>
      <td>N</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>1</td>
      <td>-0.255574</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>Partners</td>
      <td>affiliate</td>
      <td>N</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chrome</td>
      <td>iOS</td>
      <td>mobile</td>
      <td>0</td>
      <td>-0.255574</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>groups.google.com</td>
      <td>referral</td>
      <td>Category6_Path_0000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>252284</th>
      <td>Chrome</td>
      <td>Android</td>
      <td>mobile</td>
      <td>1</td>
      <td>-0.255574</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Europe</td>
      <td>Northern Europe</td>
      <td>United Kingdom</td>
      <td>youtube.com</td>
      <td>referral</td>
      <td>Category5_Path_0032</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>252285</th>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>0</td>
      <td>-0.255574</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>google</td>
      <td>organic</td>
      <td>N</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>252286</th>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>0</td>
      <td>-0.166189</td>
      <td>69.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>(direct)</td>
      <td>(none)</td>
      <td>Category1</td>
      <td>0.028986</td>
    </tr>
    <tr>
      <th>252287</th>
      <td>Android Webview</td>
      <td>Android</td>
      <td>mobile</td>
      <td>1</td>
      <td>-0.255574</td>
      <td>28.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Africa</td>
      <td>Northern Africa</td>
      <td>Egypt</td>
      <td>youtube.com</td>
      <td>referral</td>
      <td>Category2_Path_0018</td>
      <td>0.035714</td>
    </tr>
    <tr>
      <th>252288</th>
      <td>Chrome</td>
      <td>Macintosh</td>
      <td>desktop</td>
      <td>0</td>
      <td>0.101968</td>
      <td>77.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-0.042983</td>
      <td>Americas</td>
      <td>Northern America</td>
      <td>United States</td>
      <td>(direct)</td>
      <td>(none)</td>
      <td>Category1</td>
      <td>0.064935</td>
    </tr>
  </tbody>
</table>
<p>247674 rows × 16 columns</p>
</div>



```python
print(f"훈련 셋 모양 : {df_train.shape}")
print(f"타깃 모양 : {target.shape}")
```

<pre>
훈련 셋 모양 : (249362, 17)
타깃 모양 : (249362,)
</pre>
# 3. 모델 정의 및 훈련



```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df_train, target, test_size=0.1,
                                                      random_state=42)
```


```python
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

categorical_features = [
    "browser",
    "OS",
    "device",
    "subcontinent",
    "country",
    "traffic_source",
    "traffic_medium",
    "referral_path",
    "keyword",
    "continent"
]
 
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
valid_pool = Pool(data=X_valid, label=y_valid, cat_features=categorical_features)
```


```python
model = CatBoostRegressor(iterations=8000,
                          learning_rate=0.085,
                          rsm=0.5,
                          random_strength=0.5,
                          l2_leaf_reg=3.5,
                          depth=7,
                          early_stopping_rounds=500,
                          random_state=42,
                          verbose=100)
```


```python
model.fit(train_pool, eval_set=(valid_pool), verbose=2)
```

<pre>
0:	learn: 5.1874991	test: 4.9303580	best: 4.9303580 (0)	total: 23.4ms	remaining: 3m 7s
2:	learn: 4.6766737	test: 4.4174826	best: 4.4174826 (2)	total: 70.8ms	remaining: 3m 8s
4:	learn: 4.2663220	test: 4.0059270	best: 4.0059270 (4)	total: 112ms	remaining: 2m 58s
6:	learn: 3.9369545	test: 3.6757456	best: 3.6757456 (6)	total: 156ms	remaining: 2m 57s
8:	learn: 3.6934154	test: 3.4222611	best: 3.4222611 (8)	total: 201ms	remaining: 2m 58s
10:	learn: 3.5086919	test: 3.2370964	best: 3.2370964 (10)	total: 247ms	remaining: 2m 59s
12:	learn: 3.3497434	test: 3.0771650	best: 3.0771650 (12)	total: 292ms	remaining: 2m 59s
14:	learn: 3.2312509	test: 2.9551361	best: 2.9551361 (14)	total: 340ms	remaining: 3m
16:	learn: 3.1336697	test: 2.8603841	best: 2.8603841 (16)	total: 389ms	remaining: 3m 2s
18:	learn: 3.0562542	test: 2.7865161	best: 2.7865161 (18)	total: 435ms	remaining: 3m 2s
20:	learn: 2.9947078	test: 2.7258249	best: 2.7258249 (20)	total: 481ms	remaining: 3m 2s
22:	learn: 2.9374730	test: 2.6732896	best: 2.6732896 (22)	total: 527ms	remaining: 3m 2s
24:	learn: 2.8930428	test: 2.6313714	best: 2.6313714 (24)	total: 572ms	remaining: 3m 2s
26:	learn: 2.8607733	test: 2.5980631	best: 2.5980631 (26)	total: 616ms	remaining: 3m 2s
28:	learn: 2.8308691	test: 2.5715198	best: 2.5715198 (28)	total: 662ms	remaining: 3m 2s
30:	learn: 2.8082253	test: 2.5481202	best: 2.5481202 (30)	total: 709ms	remaining: 3m 2s
32:	learn: 2.7847230	test: 2.5257079	best: 2.5257079 (32)	total: 763ms	remaining: 3m 4s
34:	learn: 2.7700803	test: 2.5121261	best: 2.5121261 (34)	total: 807ms	remaining: 3m 3s
36:	learn: 2.7544485	test: 2.4958966	best: 2.4958966 (36)	total: 853ms	remaining: 3m 3s
38:	learn: 2.7388616	test: 2.4808381	best: 2.4808381 (38)	total: 897ms	remaining: 3m 3s
40:	learn: 2.7277767	test: 2.4709690	best: 2.4709690 (40)	total: 945ms	remaining: 3m 3s
42:	learn: 2.7182166	test: 2.4624388	best: 2.4624388 (42)	total: 992ms	remaining: 3m 3s
44:	learn: 2.7073883	test: 2.4517229	best: 2.4517229 (44)	total: 1.03s	remaining: 3m 2s
46:	learn: 2.6993971	test: 2.4403841	best: 2.4403841 (46)	total: 1.09s	remaining: 3m 4s
48:	learn: 2.6890374	test: 2.4313517	best: 2.4313517 (48)	total: 1.14s	remaining: 3m 4s
50:	learn: 2.6799587	test: 2.4234062	best: 2.4234062 (50)	total: 1.19s	remaining: 3m 4s
52:	learn: 2.6768308	test: 2.4187579	best: 2.4187579 (52)	total: 1.22s	remaining: 3m 2s
54:	learn: 2.6765119	test: 2.4148635	best: 2.4148635 (54)	total: 1.23s	remaining: 2m 57s
56:	learn: 2.6762086	test: 2.4112235	best: 2.4112235 (56)	total: 1.25s	remaining: 2m 54s
58:	learn: 2.6714370	test: 2.4061158	best: 2.4061158 (58)	total: 1.28s	remaining: 2m 52s
60:	learn: 2.6711732	test: 2.4028240	best: 2.4028240 (60)	total: 1.29s	remaining: 2m 48s
62:	learn: 2.6628702	test: 2.3951673	best: 2.3951673 (62)	total: 1.34s	remaining: 2m 49s
64:	learn: 2.6576218	test: 2.3909527	best: 2.3909527 (64)	total: 1.37s	remaining: 2m 47s
66:	learn: 2.6573914	test: 2.3880877	best: 2.3880877 (66)	total: 1.38s	remaining: 2m 43s
68:	learn: 2.6544033	test: 2.3867799	best: 2.3865715 (67)	total: 1.43s	remaining: 2m 44s
70:	learn: 2.6486339	test: 2.3810659	best: 2.3810659 (70)	total: 1.48s	remaining: 2m 45s
72:	learn: 2.6484422	test: 2.3785549	best: 2.3785549 (72)	total: 1.5s	remaining: 2m 43s
74:	learn: 2.6428898	test: 2.3729407	best: 2.3729407 (74)	total: 1.54s	remaining: 2m 42s
76:	learn: 2.6383223	test: 2.3696065	best: 2.3696065 (76)	total: 1.58s	remaining: 2m 42s
78:	learn: 2.6344432	test: 2.3660959	best: 2.3660959 (78)	total: 1.61s	remaining: 2m 41s
80:	learn: 2.6294924	test: 2.3633449	best: 2.3633449 (80)	total: 1.64s	remaining: 2m 40s
82:	learn: 2.6221263	test: 2.3590716	best: 2.3590716 (82)	total: 1.68s	remaining: 2m 40s
84:	learn: 2.6204223	test: 2.3579496	best: 2.3579496 (84)	total: 1.71s	remaining: 2m 39s
86:	learn: 2.6202729	test: 2.3569725	best: 2.3569682 (85)	total: 1.75s	remaining: 2m 38s
88:	learn: 2.6174119	test: 2.3547111	best: 2.3547111 (88)	total: 1.77s	remaining: 2m 37s
90:	learn: 2.6129981	test: 2.3505541	best: 2.3505541 (90)	total: 1.8s	remaining: 2m 36s
92:	learn: 2.6108340	test: 2.3482607	best: 2.3482607 (92)	total: 1.83s	remaining: 2m 35s
94:	learn: 2.6105602	test: 2.3474183	best: 2.3474183 (94)	total: 1.86s	remaining: 2m 34s
96:	learn: 2.6083361	test: 2.3462216	best: 2.3462216 (96)	total: 1.88s	remaining: 2m 33s
98:	learn: 2.6048634	test: 2.3430902	best: 2.3430902 (98)	total: 1.93s	remaining: 2m 33s
100:	learn: 2.6047670	test: 2.3416528	best: 2.3416528 (100)	total: 1.94s	remaining: 2m 32s
102:	learn: 2.6043377	test: 2.3409883	best: 2.3409671 (101)	total: 1.97s	remaining: 2m 31s
104:	learn: 2.6026190	test: 2.3395871	best: 2.3395871 (104)	total: 2s	remaining: 2m 30s
106:	learn: 2.6007269	test: 2.3383419	best: 2.3383419 (106)	total: 2.03s	remaining: 2m 29s
108:	learn: 2.5956989	test: 2.3352980	best: 2.3352980 (108)	total: 2.08s	remaining: 2m 30s
110:	learn: 2.5905370	test: 2.3320982	best: 2.3320982 (110)	total: 2.12s	remaining: 2m 30s
112:	learn: 2.5871807	test: 2.3289604	best: 2.3289604 (112)	total: 2.14s	remaining: 2m 29s
114:	learn: 2.5820283	test: 2.3239223	best: 2.3239223 (114)	total: 2.19s	remaining: 2m 30s
116:	learn: 2.5787956	test: 2.3228791	best: 2.3228791 (116)	total: 2.24s	remaining: 2m 31s
118:	learn: 2.5759905	test: 2.3207180	best: 2.3207180 (118)	total: 2.27s	remaining: 2m 30s
120:	learn: 2.5735322	test: 2.3183318	best: 2.3183318 (120)	total: 2.31s	remaining: 2m 30s
122:	learn: 2.5713782	test: 2.3179404	best: 2.3179404 (122)	total: 2.34s	remaining: 2m 30s
124:	learn: 2.5683777	test: 2.3163545	best: 2.3163545 (124)	total: 2.38s	remaining: 2m 30s
126:	learn: 2.5671951	test: 2.3152082	best: 2.3152082 (126)	total: 2.42s	remaining: 2m 29s
128:	learn: 2.5657242	test: 2.3140408	best: 2.3140408 (128)	total: 2.46s	remaining: 2m 30s
130:	learn: 2.5610502	test: 2.3113002	best: 2.3113002 (130)	total: 2.53s	remaining: 2m 31s
132:	learn: 2.5607361	test: 2.3110040	best: 2.3110040 (132)	total: 2.56s	remaining: 2m 31s
134:	learn: 2.5570366	test: 2.3085832	best: 2.3085832 (134)	total: 2.59s	remaining: 2m 30s
136:	learn: 2.5544370	test: 2.3054974	best: 2.3054974 (136)	total: 2.62s	remaining: 2m 30s
138:	learn: 2.5509386	test: 2.3044425	best: 2.3044425 (138)	total: 2.67s	remaining: 2m 30s
140:	learn: 2.5493049	test: 2.3031927	best: 2.3031927 (140)	total: 2.72s	remaining: 2m 31s
142:	learn: 2.5477202	test: 2.3027117	best: 2.3027117 (142)	total: 2.77s	remaining: 2m 32s
144:	learn: 2.5435416	test: 2.3011248	best: 2.3011248 (144)	total: 2.82s	remaining: 2m 32s
146:	learn: 2.5407339	test: 2.2986831	best: 2.2986831 (146)	total: 2.86s	remaining: 2m 32s
148:	learn: 2.5387962	test: 2.2975849	best: 2.2975849 (148)	total: 2.91s	remaining: 2m 33s
150:	learn: 2.5366440	test: 2.2960887	best: 2.2960887 (150)	total: 2.96s	remaining: 2m 33s
152:	learn: 2.5356000	test: 2.2954113	best: 2.2954113 (152)	total: 3.01s	remaining: 2m 34s
154:	learn: 2.5336334	test: 2.2944607	best: 2.2944607 (154)	total: 3.06s	remaining: 2m 34s
156:	learn: 2.5307119	test: 2.2918117	best: 2.2918117 (156)	total: 3.09s	remaining: 2m 34s
158:	learn: 2.5287766	test: 2.2897329	best: 2.2897329 (158)	total: 3.13s	remaining: 2m 34s
160:	learn: 2.5285274	test: 2.2894311	best: 2.2894311 (160)	total: 3.18s	remaining: 2m 34s
162:	learn: 2.5283238	test: 2.2892884	best: 2.2892884 (162)	total: 3.22s	remaining: 2m 34s
164:	learn: 2.5267977	test: 2.2879826	best: 2.2879826 (164)	total: 3.26s	remaining: 2m 34s
166:	learn: 2.5242025	test: 2.2866785	best: 2.2866785 (166)	total: 3.31s	remaining: 2m 35s
168:	learn: 2.5239242	test: 2.2863345	best: 2.2863345 (168)	total: 3.35s	remaining: 2m 35s
170:	learn: 2.5233849	test: 2.2861144	best: 2.2861144 (170)	total: 3.4s	remaining: 2m 35s
172:	learn: 2.5220089	test: 2.2855919	best: 2.2855919 (172)	total: 3.45s	remaining: 2m 36s
174:	learn: 2.5209477	test: 2.2847960	best: 2.2847960 (174)	total: 3.51s	remaining: 2m 36s
176:	learn: 2.5208295	test: 2.2842804	best: 2.2842804 (176)	total: 3.54s	remaining: 2m 36s
178:	learn: 2.5206344	test: 2.2837950	best: 2.2837950 (178)	total: 3.57s	remaining: 2m 35s
180:	learn: 2.5201225	test: 2.2832253	best: 2.2832253 (180)	total: 3.61s	remaining: 2m 35s
182:	learn: 2.5198832	test: 2.2829784	best: 2.2829780 (181)	total: 3.65s	remaining: 2m 36s
184:	learn: 2.5182218	test: 2.2811490	best: 2.2811490 (184)	total: 3.7s	remaining: 2m 36s
186:	learn: 2.5165064	test: 2.2804433	best: 2.2804433 (186)	total: 3.75s	remaining: 2m 36s
188:	learn: 2.5158163	test: 2.2796778	best: 2.2796777 (187)	total: 3.79s	remaining: 2m 36s
190:	learn: 2.5127183	test: 2.2791821	best: 2.2791821 (190)	total: 3.84s	remaining: 2m 37s
192:	learn: 2.5103271	test: 2.2782300	best: 2.2782300 (192)	total: 3.91s	remaining: 2m 38s
194:	learn: 2.5099997	test: 2.2778398	best: 2.2778398 (194)	total: 3.96s	remaining: 2m 38s
196:	learn: 2.5084591	test: 2.2773228	best: 2.2773228 (196)	total: 4.02s	remaining: 2m 39s
198:	learn: 2.5069498	test: 2.2771714	best: 2.2771714 (198)	total: 4.08s	remaining: 2m 40s
200:	learn: 2.5046785	test: 2.2764751	best: 2.2764751 (200)	total: 4.15s	remaining: 2m 40s
202:	learn: 2.5022836	test: 2.2761665	best: 2.2761665 (202)	total: 4.2s	remaining: 2m 41s
204:	learn: 2.5002393	test: 2.2738583	best: 2.2738583 (204)	total: 4.26s	remaining: 2m 41s
206:	learn: 2.4985770	test: 2.2737527	best: 2.2737270 (205)	total: 4.31s	remaining: 2m 42s
208:	learn: 2.4974840	test: 2.2726802	best: 2.2726802 (208)	total: 4.36s	remaining: 2m 42s
210:	learn: 2.4963836	test: 2.2720598	best: 2.2720598 (210)	total: 4.42s	remaining: 2m 43s
212:	learn: 2.4955523	test: 2.2720563	best: 2.2719072 (211)	total: 4.47s	remaining: 2m 43s
214:	learn: 2.4948047	test: 2.2714350	best: 2.2714350 (214)	total: 4.53s	remaining: 2m 43s
216:	learn: 2.4937282	test: 2.2709620	best: 2.2709620 (216)	total: 4.59s	remaining: 2m 44s
218:	learn: 2.4918784	test: 2.2701217	best: 2.2701217 (218)	total: 4.63s	remaining: 2m 44s
220:	learn: 2.4917794	test: 2.2701188	best: 2.2701188 (220)	total: 4.68s	remaining: 2m 44s
222:	learn: 2.4896544	test: 2.2691832	best: 2.2691832 (222)	total: 4.75s	remaining: 2m 45s
224:	learn: 2.4888957	test: 2.2690234	best: 2.2689980 (223)	total: 4.8s	remaining: 2m 45s
226:	learn: 2.4886363	test: 2.2689308	best: 2.2689308 (226)	total: 4.85s	remaining: 2m 46s
228:	learn: 2.4867244	test: 2.2678060	best: 2.2678060 (228)	total: 4.92s	remaining: 2m 46s
230:	learn: 2.4858961	test: 2.2671951	best: 2.2671951 (230)	total: 4.97s	remaining: 2m 47s
232:	learn: 2.4846620	test: 2.2667271	best: 2.2667271 (232)	total: 5.02s	remaining: 2m 47s
234:	learn: 2.4829557	test: 2.2655235	best: 2.2655235 (234)	total: 5.08s	remaining: 2m 47s
236:	learn: 2.4823968	test: 2.2655999	best: 2.2655211 (235)	total: 5.13s	remaining: 2m 48s
238:	learn: 2.4804118	test: 2.2658865	best: 2.2655211 (235)	total: 5.18s	remaining: 2m 48s
240:	learn: 2.4797489	test: 2.2657270	best: 2.2655211 (235)	total: 5.24s	remaining: 2m 48s
242:	learn: 2.4795471	test: 2.2656964	best: 2.2655211 (235)	total: 5.29s	remaining: 2m 48s
244:	learn: 2.4792410	test: 2.2655130	best: 2.2655130 (244)	total: 5.35s	remaining: 2m 49s
246:	learn: 2.4783770	test: 2.2652081	best: 2.2652081 (246)	total: 5.4s	remaining: 2m 49s
248:	learn: 2.4764413	test: 2.2648358	best: 2.2648358 (248)	total: 5.45s	remaining: 2m 49s
250:	learn: 2.4758786	test: 2.2646002	best: 2.2643801 (249)	total: 5.5s	remaining: 2m 49s
252:	learn: 2.4756695	test: 2.2644312	best: 2.2643801 (249)	total: 5.56s	remaining: 2m 50s
254:	learn: 2.4735068	test: 2.2636483	best: 2.2636483 (254)	total: 5.61s	remaining: 2m 50s
256:	learn: 2.4731159	test: 2.2632512	best: 2.2632512 (256)	total: 5.67s	remaining: 2m 50s
258:	learn: 2.4725886	test: 2.2633988	best: 2.2632512 (256)	total: 5.72s	remaining: 2m 50s
260:	learn: 2.4720490	test: 2.2633240	best: 2.2632512 (256)	total: 5.76s	remaining: 2m 50s
262:	learn: 2.4717688	test: 2.2635257	best: 2.2632512 (256)	total: 5.82s	remaining: 2m 51s
264:	learn: 2.4700181	test: 2.2623302	best: 2.2623302 (264)	total: 5.87s	remaining: 2m 51s
266:	learn: 2.4692172	test: 2.2620779	best: 2.2620779 (266)	total: 5.92s	remaining: 2m 51s
268:	learn: 2.4679859	test: 2.2613685	best: 2.2613685 (268)	total: 5.98s	remaining: 2m 51s
270:	learn: 2.4673183	test: 2.2612073	best: 2.2612073 (270)	total: 6.03s	remaining: 2m 51s
272:	learn: 2.4672485	test: 2.2609612	best: 2.2609612 (272)	total: 6.08s	remaining: 2m 52s
274:	learn: 2.4660569	test: 2.2606700	best: 2.2606700 (274)	total: 6.14s	remaining: 2m 52s
276:	learn: 2.4649971	test: 2.2600751	best: 2.2600751 (276)	total: 6.19s	remaining: 2m 52s
278:	learn: 2.4640494	test: 2.2595003	best: 2.2595003 (278)	total: 6.25s	remaining: 2m 52s
280:	learn: 2.4630362	test: 2.2591917	best: 2.2591742 (279)	total: 6.3s	remaining: 2m 53s
282:	learn: 2.4629271	test: 2.2591135	best: 2.2591127 (281)	total: 6.35s	remaining: 2m 53s
284:	learn: 2.4610782	test: 2.2585736	best: 2.2585492 (283)	total: 6.4s	remaining: 2m 53s
286:	learn: 2.4609284	test: 2.2584701	best: 2.2584701 (286)	total: 6.45s	remaining: 2m 53s
288:	learn: 2.4607641	test: 2.2583192	best: 2.2583192 (288)	total: 6.49s	remaining: 2m 53s
290:	learn: 2.4606334	test: 2.2580854	best: 2.2580854 (290)	total: 6.53s	remaining: 2m 53s
292:	learn: 2.4603769	test: 2.2578050	best: 2.2578050 (292)	total: 6.59s	remaining: 2m 53s
294:	learn: 2.4584886	test: 2.2557638	best: 2.2557638 (294)	total: 6.64s	remaining: 2m 53s
296:	learn: 2.4564996	test: 2.2555831	best: 2.2555831 (296)	total: 6.69s	remaining: 2m 53s
298:	learn: 2.4558245	test: 2.2550906	best: 2.2550906 (298)	total: 6.75s	remaining: 2m 53s
300:	learn: 2.4546435	test: 2.2558515	best: 2.2549983 (299)	total: 6.8s	remaining: 2m 54s
302:	learn: 2.4542646	test: 2.2554703	best: 2.2549983 (299)	total: 6.85s	remaining: 2m 54s
304:	learn: 2.4538978	test: 2.2560009	best: 2.2549983 (299)	total: 6.9s	remaining: 2m 54s
306:	learn: 2.4528518	test: 2.2558934	best: 2.2549983 (299)	total: 6.95s	remaining: 2m 54s
308:	learn: 2.4511097	test: 2.2550522	best: 2.2549983 (299)	total: 7s	remaining: 2m 54s
310:	learn: 2.4492193	test: 2.2542403	best: 2.2542403 (310)	total: 7.05s	remaining: 2m 54s
312:	learn: 2.4487293	test: 2.2542411	best: 2.2542193 (311)	total: 7.1s	remaining: 2m 54s
314:	learn: 2.4484910	test: 2.2540620	best: 2.2540620 (314)	total: 7.15s	remaining: 2m 54s
316:	learn: 2.4468999	test: 2.2539197	best: 2.2539197 (316)	total: 7.2s	remaining: 2m 54s
318:	learn: 2.4443056	test: 2.2535426	best: 2.2535119 (317)	total: 7.25s	remaining: 2m 54s
320:	learn: 2.4441073	test: 2.2534825	best: 2.2534066 (319)	total: 7.31s	remaining: 2m 54s
322:	learn: 2.4414953	test: 2.2522101	best: 2.2522101 (322)	total: 7.35s	remaining: 2m 54s
324:	learn: 2.4395114	test: 2.2519420	best: 2.2519420 (324)	total: 7.4s	remaining: 2m 54s
326:	learn: 2.4361447	test: 2.2513291	best: 2.2513291 (326)	total: 7.45s	remaining: 2m 54s
328:	learn: 2.4350712	test: 2.2508014	best: 2.2508014 (328)	total: 7.51s	remaining: 2m 55s
330:	learn: 2.4345037	test: 2.2508121	best: 2.2508014 (328)	total: 7.57s	remaining: 2m 55s
332:	learn: 2.4339272	test: 2.2500667	best: 2.2500667 (332)	total: 7.62s	remaining: 2m 55s
334:	learn: 2.4338298	test: 2.2498019	best: 2.2498019 (334)	total: 7.67s	remaining: 2m 55s
336:	learn: 2.4324422	test: 2.2501460	best: 2.2498019 (334)	total: 7.72s	remaining: 2m 55s
338:	learn: 2.4321329	test: 2.2500617	best: 2.2498019 (334)	total: 7.78s	remaining: 2m 55s
340:	learn: 2.4318944	test: 2.2499402	best: 2.2498019 (334)	total: 7.83s	remaining: 2m 55s
342:	learn: 2.4314170	test: 2.2499899	best: 2.2498019 (334)	total: 7.88s	remaining: 2m 55s
344:	learn: 2.4312974	test: 2.2498874	best: 2.2498019 (334)	total: 7.94s	remaining: 2m 56s
346:	learn: 2.4306439	test: 2.2496786	best: 2.2496786 (346)	total: 7.99s	remaining: 2m 56s
348:	learn: 2.4287204	test: 2.2496941	best: 2.2496786 (346)	total: 8.05s	remaining: 2m 56s
350:	learn: 2.4272549	test: 2.2496547	best: 2.2496547 (350)	total: 8.11s	remaining: 2m 56s
352:	learn: 2.4267652	test: 2.2491247	best: 2.2491247 (352)	total: 8.17s	remaining: 2m 56s
354:	learn: 2.4261835	test: 2.2491968	best: 2.2490065 (353)	total: 8.22s	remaining: 2m 57s
356:	learn: 2.4257522	test: 2.2490846	best: 2.2490065 (353)	total: 8.28s	remaining: 2m 57s
358:	learn: 2.4254221	test: 2.2487803	best: 2.2487803 (358)	total: 8.33s	remaining: 2m 57s
360:	learn: 2.4246722	test: 2.2488571	best: 2.2487590 (359)	total: 8.39s	remaining: 2m 57s
362:	learn: 2.4237240	test: 2.2493952	best: 2.2487590 (359)	total: 8.44s	remaining: 2m 57s
364:	learn: 2.4225704	test: 2.2492264	best: 2.2487590 (359)	total: 8.49s	remaining: 2m 57s
366:	learn: 2.4217139	test: 2.2490212	best: 2.2487590 (359)	total: 8.55s	remaining: 2m 57s
368:	learn: 2.4212840	test: 2.2491633	best: 2.2487590 (359)	total: 8.62s	remaining: 2m 58s
370:	learn: 2.4205884	test: 2.2488144	best: 2.2487590 (359)	total: 8.66s	remaining: 2m 58s
372:	learn: 2.4194624	test: 2.2484601	best: 2.2484601 (372)	total: 8.73s	remaining: 2m 58s
374:	learn: 2.4186180	test: 2.2480954	best: 2.2480954 (374)	total: 8.8s	remaining: 2m 58s
376:	learn: 2.4179286	test: 2.2476465	best: 2.2475405 (375)	total: 8.85s	remaining: 2m 58s
378:	learn: 2.4172854	test: 2.2474539	best: 2.2474539 (378)	total: 8.89s	remaining: 2m 58s
380:	learn: 2.4172627	test: 2.2473336	best: 2.2473336 (380)	total: 8.94s	remaining: 2m 58s
382:	learn: 2.4157437	test: 2.2466544	best: 2.2466544 (382)	total: 8.99s	remaining: 2m 58s
384:	learn: 2.4157251	test: 2.2465940	best: 2.2465940 (384)	total: 9.04s	remaining: 2m 58s
386:	learn: 2.4151443	test: 2.2461480	best: 2.2461480 (386)	total: 9.1s	remaining: 2m 58s
388:	learn: 2.4148592	test: 2.2463264	best: 2.2461447 (387)	total: 9.14s	remaining: 2m 58s
390:	learn: 2.4130188	test: 2.2462668	best: 2.2461447 (387)	total: 9.2s	remaining: 2m 58s
392:	learn: 2.4128329	test: 2.2462837	best: 2.2461447 (387)	total: 9.25s	remaining: 2m 59s
394:	learn: 2.4119105	test: 2.2461581	best: 2.2461447 (387)	total: 9.3s	remaining: 2m 59s
396:	learn: 2.4116854	test: 2.2460728	best: 2.2460728 (396)	total: 9.37s	remaining: 2m 59s
398:	learn: 2.4107402	test: 2.2456520	best: 2.2456520 (398)	total: 9.41s	remaining: 2m 59s
400:	learn: 2.4104957	test: 2.2456082	best: 2.2456082 (400)	total: 9.47s	remaining: 2m 59s
402:	learn: 2.4090832	test: 2.2446913	best: 2.2446913 (402)	total: 9.53s	remaining: 2m 59s
404:	learn: 2.4076070	test: 2.2437868	best: 2.2437868 (404)	total: 9.58s	remaining: 2m 59s
406:	learn: 2.4064068	test: 2.2440687	best: 2.2435949 (405)	total: 9.63s	remaining: 2m 59s
408:	learn: 2.4056001	test: 2.2437884	best: 2.2435949 (405)	total: 9.68s	remaining: 2m 59s
410:	learn: 2.4047505	test: 2.2436222	best: 2.2435949 (405)	total: 9.74s	remaining: 2m 59s
412:	learn: 2.4019977	test: 2.2423315	best: 2.2423315 (412)	total: 9.79s	remaining: 2m 59s
414:	learn: 2.4012710	test: 2.2417488	best: 2.2417488 (414)	total: 9.83s	remaining: 2m 59s
416:	learn: 2.4008132	test: 2.2415775	best: 2.2415775 (416)	total: 9.88s	remaining: 2m 59s
418:	learn: 2.4001993	test: 2.2413489	best: 2.2413489 (418)	total: 9.93s	remaining: 2m 59s
420:	learn: 2.3990846	test: 2.2411800	best: 2.2411800 (420)	total: 9.98s	remaining: 2m 59s
422:	learn: 2.3981900	test: 2.2411539	best: 2.2411539 (422)	total: 10s	remaining: 2m 59s
424:	learn: 2.3972837	test: 2.2406524	best: 2.2406524 (424)	total: 10.1s	remaining: 2m 59s
426:	learn: 2.3961623	test: 2.2402000	best: 2.2402000 (426)	total: 10.1s	remaining: 2m 59s
428:	learn: 2.3955708	test: 2.2401927	best: 2.2401927 (428)	total: 10.2s	remaining: 2m 59s
430:	learn: 2.3938633	test: 2.2405208	best: 2.2401927 (428)	total: 10.2s	remaining: 2m 59s
432:	learn: 2.3931207	test: 2.2401992	best: 2.2398438 (431)	total: 10.3s	remaining: 2m 59s
434:	learn: 2.3915981	test: 2.2391189	best: 2.2391189 (434)	total: 10.3s	remaining: 2m 59s
436:	learn: 2.3906860	test: 2.2387579	best: 2.2387579 (436)	total: 10.4s	remaining: 3m
438:	learn: 2.3903841	test: 2.2387101	best: 2.2387101 (438)	total: 10.5s	remaining: 3m
440:	learn: 2.3894458	test: 2.2386975	best: 2.2384919 (439)	total: 10.5s	remaining: 3m
442:	learn: 2.3892317	test: 2.2386055	best: 2.2384919 (439)	total: 10.6s	remaining: 3m
444:	learn: 2.3881780	test: 2.2380969	best: 2.2380969 (444)	total: 10.6s	remaining: 3m
446:	learn: 2.3873877	test: 2.2380647	best: 2.2380531 (445)	total: 10.7s	remaining: 3m
448:	learn: 2.3868641	test: 2.2380512	best: 2.2380512 (448)	total: 10.7s	remaining: 3m
450:	learn: 2.3863887	test: 2.2379738	best: 2.2379738 (450)	total: 10.8s	remaining: 3m
452:	learn: 2.3858547	test: 2.2374220	best: 2.2374220 (452)	total: 10.8s	remaining: 3m
454:	learn: 2.3855603	test: 2.2375281	best: 2.2374220 (452)	total: 10.9s	remaining: 3m
456:	learn: 2.3854435	test: 2.2375141	best: 2.2374220 (452)	total: 10.9s	remaining: 3m
458:	learn: 2.3852633	test: 2.2378010	best: 2.2374220 (452)	total: 11s	remaining: 3m
460:	learn: 2.3852285	test: 2.2377404	best: 2.2374220 (452)	total: 11s	remaining: 3m
462:	learn: 2.3845461	test: 2.2373594	best: 2.2373594 (462)	total: 11.1s	remaining: 3m
464:	learn: 2.3832706	test: 2.2372757	best: 2.2372757 (464)	total: 11.1s	remaining: 3m
466:	learn: 2.3827583	test: 2.2372910	best: 2.2372757 (464)	total: 11.2s	remaining: 3m
468:	learn: 2.3822128	test: 2.2372578	best: 2.2372578 (468)	total: 11.2s	remaining: 3m
470:	learn: 2.3815273	test: 2.2373787	best: 2.2372578 (468)	total: 11.3s	remaining: 3m
472:	learn: 2.3803515	test: 2.2364977	best: 2.2364383 (471)	total: 11.3s	remaining: 3m
474:	learn: 2.3796277	test: 2.2365000	best: 2.2364383 (471)	total: 11.4s	remaining: 3m
476:	learn: 2.3787847	test: 2.2362461	best: 2.2362461 (476)	total: 11.4s	remaining: 3m
478:	learn: 2.3778204	test: 2.2356522	best: 2.2356522 (478)	total: 11.5s	remaining: 3m
480:	learn: 2.3769927	test: 2.2352946	best: 2.2352946 (480)	total: 11.6s	remaining: 3m
482:	learn: 2.3750964	test: 2.2353166	best: 2.2350671 (481)	total: 11.6s	remaining: 3m
484:	learn: 2.3730392	test: 2.2349805	best: 2.2348638 (483)	total: 11.6s	remaining: 3m
486:	learn: 2.3730039	test: 2.2348920	best: 2.2348638 (483)	total: 11.7s	remaining: 3m
488:	learn: 2.3717782	test: 2.2349068	best: 2.2348638 (483)	total: 11.8s	remaining: 3m
490:	learn: 2.3711511	test: 2.2355467	best: 2.2348638 (483)	total: 11.8s	remaining: 3m
492:	learn: 2.3706965	test: 2.2357409	best: 2.2348638 (483)	total: 11.9s	remaining: 3m
494:	learn: 2.3704901	test: 2.2358121	best: 2.2348638 (483)	total: 11.9s	remaining: 3m
496:	learn: 2.3700889	test: 2.2353923	best: 2.2348638 (483)	total: 12s	remaining: 3m
498:	learn: 2.3693440	test: 2.2350538	best: 2.2348638 (483)	total: 12s	remaining: 3m
500:	learn: 2.3690138	test: 2.2346429	best: 2.2346429 (500)	total: 12.1s	remaining: 3m
502:	learn: 2.3680278	test: 2.2347672	best: 2.2346429 (500)	total: 12.1s	remaining: 3m
504:	learn: 2.3657644	test: 2.2344348	best: 2.2344348 (504)	total: 12.2s	remaining: 3m
506:	learn: 2.3654161	test: 2.2345957	best: 2.2344348 (504)	total: 12.2s	remaining: 3m
508:	learn: 2.3648096	test: 2.2344307	best: 2.2344307 (508)	total: 12.3s	remaining: 3m
510:	learn: 2.3636692	test: 2.2341041	best: 2.2341041 (510)	total: 12.3s	remaining: 3m
512:	learn: 2.3634510	test: 2.2340220	best: 2.2340220 (512)	total: 12.4s	remaining: 3m
514:	learn: 2.3626547	test: 2.2336152	best: 2.2336152 (514)	total: 12.4s	remaining: 3m
516:	learn: 2.3621324	test: 2.2335043	best: 2.2335043 (516)	total: 12.5s	remaining: 3m
518:	learn: 2.3616370	test: 2.2330759	best: 2.2330759 (518)	total: 12.5s	remaining: 3m
520:	learn: 2.3611660	test: 2.2327378	best: 2.2326682 (519)	total: 12.6s	remaining: 3m
522:	learn: 2.3608360	test: 2.2327087	best: 2.2326682 (519)	total: 12.7s	remaining: 3m
524:	learn: 2.3597928	test: 2.2325671	best: 2.2325671 (524)	total: 12.7s	remaining: 3m
526:	learn: 2.3593177	test: 2.2324268	best: 2.2324268 (526)	total: 12.8s	remaining: 3m
528:	learn: 2.3584844	test: 2.2320850	best: 2.2320850 (528)	total: 12.8s	remaining: 3m
530:	learn: 2.3580886	test: 2.2320826	best: 2.2320775 (529)	total: 12.9s	remaining: 3m
532:	learn: 2.3568863	test: 2.2322588	best: 2.2318354 (531)	total: 12.9s	remaining: 3m 1s
534:	learn: 2.3560193	test: 2.2319541	best: 2.2318354 (531)	total: 13s	remaining: 3m 1s
536:	learn: 2.3548973	test: 2.2312932	best: 2.2312932 (536)	total: 13s	remaining: 3m 1s
538:	learn: 2.3544579	test: 2.2314955	best: 2.2312932 (536)	total: 13.1s	remaining: 3m 1s
540:	learn: 2.3537838	test: 2.2313728	best: 2.2312932 (536)	total: 13.1s	remaining: 3m 1s
542:	learn: 2.3532141	test: 2.2309504	best: 2.2309504 (542)	total: 13.2s	remaining: 3m 1s
544:	learn: 2.3530054	test: 2.2308796	best: 2.2308796 (544)	total: 13.2s	remaining: 3m 1s
546:	learn: 2.3525782	test: 2.2307196	best: 2.2307196 (546)	total: 13.3s	remaining: 3m 1s
548:	learn: 2.3520971	test: 2.2308832	best: 2.2307196 (546)	total: 13.4s	remaining: 3m 1s
550:	learn: 2.3520530	test: 2.2307553	best: 2.2307196 (546)	total: 13.4s	remaining: 3m 1s
552:	learn: 2.3511179	test: 2.2305798	best: 2.2305798 (552)	total: 13.4s	remaining: 3m 1s
554:	learn: 2.3510504	test: 2.2305686	best: 2.2305686 (554)	total: 13.5s	remaining: 3m 1s
556:	learn: 2.3499476	test: 2.2301061	best: 2.2301061 (556)	total: 13.6s	remaining: 3m 1s
558:	learn: 2.3495767	test: 2.2300418	best: 2.2300185 (557)	total: 13.6s	remaining: 3m 1s
560:	learn: 2.3487580	test: 2.2297311	best: 2.2297311 (560)	total: 13.7s	remaining: 3m 1s
562:	learn: 2.3482925	test: 2.2296701	best: 2.2296701 (562)	total: 13.7s	remaining: 3m
564:	learn: 2.3478271	test: 2.2295669	best: 2.2295669 (564)	total: 13.7s	remaining: 3m
566:	learn: 2.3476335	test: 2.2296756	best: 2.2295669 (564)	total: 13.8s	remaining: 3m
568:	learn: 2.3472235	test: 2.2299339	best: 2.2295669 (564)	total: 13.8s	remaining: 3m
570:	learn: 2.3464495	test: 2.2292165	best: 2.2292165 (570)	total: 13.9s	remaining: 3m
572:	learn: 2.3460159	test: 2.2286159	best: 2.2286159 (572)	total: 14s	remaining: 3m
574:	learn: 2.3456851	test: 2.2288230	best: 2.2286077 (573)	total: 14s	remaining: 3m
576:	learn: 2.3452273	test: 2.2290078	best: 2.2286077 (573)	total: 14.1s	remaining: 3m
578:	learn: 2.3449119	test: 2.2286913	best: 2.2286077 (573)	total: 14.1s	remaining: 3m
580:	learn: 2.3440894	test: 2.2281282	best: 2.2281282 (580)	total: 14.2s	remaining: 3m
582:	learn: 2.3435036	test: 2.2280733	best: 2.2280733 (582)	total: 14.2s	remaining: 3m
584:	learn: 2.3432403	test: 2.2279708	best: 2.2279708 (584)	total: 14.3s	remaining: 3m
586:	learn: 2.3425396	test: 2.2280863	best: 2.2279708 (584)	total: 14.3s	remaining: 3m
588:	learn: 2.3423150	test: 2.2285059	best: 2.2279708 (584)	total: 14.4s	remaining: 3m
590:	learn: 2.3419509	test: 2.2288702	best: 2.2279708 (584)	total: 14.4s	remaining: 3m
592:	learn: 2.3409809	test: 2.2285510	best: 2.2279708 (584)	total: 14.5s	remaining: 3m
594:	learn: 2.3396552	test: 2.2281258	best: 2.2279708 (584)	total: 14.5s	remaining: 3m
596:	learn: 2.3392486	test: 2.2283222	best: 2.2279708 (584)	total: 14.6s	remaining: 3m 1s
598:	learn: 2.3388133	test: 2.2279464	best: 2.2279464 (598)	total: 14.7s	remaining: 3m 1s
600:	learn: 2.3381775	test: 2.2277696	best: 2.2277696 (600)	total: 14.7s	remaining: 3m 1s
602:	learn: 2.3362118	test: 2.2264878	best: 2.2264878 (602)	total: 14.8s	remaining: 3m 1s
604:	learn: 2.3359704	test: 2.2264747	best: 2.2264747 (604)	total: 14.8s	remaining: 3m 1s
606:	learn: 2.3351252	test: 2.2265535	best: 2.2262850 (605)	total: 14.9s	remaining: 3m 1s
608:	learn: 2.3337165	test: 2.2264253	best: 2.2262850 (605)	total: 14.9s	remaining: 3m 1s
610:	learn: 2.3334547	test: 2.2262095	best: 2.2262095 (610)	total: 15s	remaining: 3m 1s
612:	learn: 2.3320794	test: 2.2248997	best: 2.2248997 (612)	total: 15s	remaining: 3m 1s
614:	learn: 2.3316726	test: 2.2247515	best: 2.2247515 (614)	total: 15.1s	remaining: 3m
616:	learn: 2.3313558	test: 2.2247004	best: 2.2246981 (615)	total: 15.1s	remaining: 3m
618:	learn: 2.3309735	test: 2.2244678	best: 2.2244678 (618)	total: 15.2s	remaining: 3m
620:	learn: 2.3309021	test: 2.2242972	best: 2.2242972 (620)	total: 15.2s	remaining: 3m
622:	learn: 2.3303195	test: 2.2240865	best: 2.2240865 (622)	total: 15.3s	remaining: 3m
624:	learn: 2.3300318	test: 2.2239489	best: 2.2239489 (624)	total: 15.3s	remaining: 3m
626:	learn: 2.3295242	test: 2.2235996	best: 2.2234778 (625)	total: 15.4s	remaining: 3m
628:	learn: 2.3287937	test: 2.2237482	best: 2.2234778 (625)	total: 15.4s	remaining: 3m
630:	learn: 2.3284774	test: 2.2236563	best: 2.2234778 (625)	total: 15.5s	remaining: 3m
632:	learn: 2.3282158	test: 2.2236185	best: 2.2234778 (625)	total: 15.5s	remaining: 3m
634:	learn: 2.3278931	test: 2.2236845	best: 2.2234778 (625)	total: 15.6s	remaining: 3m
636:	learn: 2.3272780	test: 2.2238330	best: 2.2234778 (625)	total: 15.6s	remaining: 3m
638:	learn: 2.3268265	test: 2.2237823	best: 2.2234778 (625)	total: 15.7s	remaining: 3m
640:	learn: 2.3263672	test: 2.2228507	best: 2.2228507 (640)	total: 15.7s	remaining: 3m
642:	learn: 2.3261030	test: 2.2228194	best: 2.2228194 (642)	total: 15.8s	remaining: 3m
644:	learn: 2.3254372	test: 2.2232627	best: 2.2228194 (642)	total: 15.8s	remaining: 3m
646:	learn: 2.3247401	test: 2.2227203	best: 2.2227203 (646)	total: 15.9s	remaining: 3m
648:	learn: 2.3225069	test: 2.2225221	best: 2.2225221 (648)	total: 15.9s	remaining: 3m
650:	learn: 2.3212022	test: 2.2226177	best: 2.2222920 (649)	total: 16s	remaining: 3m
652:	learn: 2.3205795	test: 2.2224835	best: 2.2222920 (649)	total: 16s	remaining: 3m
654:	learn: 2.3205023	test: 2.2225267	best: 2.2222920 (649)	total: 16.1s	remaining: 3m
656:	learn: 2.3201721	test: 2.2223855	best: 2.2222920 (649)	total: 16.2s	remaining: 3m
658:	learn: 2.3198732	test: 2.2227683	best: 2.2222920 (649)	total: 16.2s	remaining: 3m
660:	learn: 2.3193440	test: 2.2227304	best: 2.2222920 (649)	total: 16.3s	remaining: 3m
662:	learn: 2.3176741	test: 2.2227526	best: 2.2222920 (649)	total: 16.3s	remaining: 3m
664:	learn: 2.3164305	test: 2.2222373	best: 2.2222373 (664)	total: 16.4s	remaining: 3m
666:	learn: 2.3162228	test: 2.2223226	best: 2.2222373 (664)	total: 16.4s	remaining: 3m
668:	learn: 2.3156826	test: 2.2222785	best: 2.2222373 (664)	total: 16.5s	remaining: 3m
670:	learn: 2.3140985	test: 2.2221906	best: 2.2221906 (670)	total: 16.5s	remaining: 3m
672:	learn: 2.3132172	test: 2.2228364	best: 2.2221906 (670)	total: 16.6s	remaining: 3m
674:	learn: 2.3130790	test: 2.2228662	best: 2.2221906 (670)	total: 16.6s	remaining: 3m
676:	learn: 2.3129274	test: 2.2228495	best: 2.2221906 (670)	total: 16.7s	remaining: 3m
678:	learn: 2.3124281	test: 2.2234172	best: 2.2221906 (670)	total: 16.7s	remaining: 3m
680:	learn: 2.3117681	test: 2.2234645	best: 2.2221906 (670)	total: 16.8s	remaining: 3m
682:	learn: 2.3103635	test: 2.2232698	best: 2.2221906 (670)	total: 16.8s	remaining: 3m
684:	learn: 2.3097308	test: 2.2231605	best: 2.2221906 (670)	total: 16.9s	remaining: 3m
686:	learn: 2.3091484	test: 2.2231037	best: 2.2221906 (670)	total: 16.9s	remaining: 3m
688:	learn: 2.3087696	test: 2.2230606	best: 2.2221906 (670)	total: 17s	remaining: 3m
690:	learn: 2.3086166	test: 2.2230161	best: 2.2221906 (670)	total: 17.1s	remaining: 3m
692:	learn: 2.3084860	test: 2.2230912	best: 2.2221906 (670)	total: 17.1s	remaining: 3m
694:	learn: 2.3081745	test: 2.2229590	best: 2.2221906 (670)	total: 17.2s	remaining: 3m
696:	learn: 2.3067621	test: 2.2234554	best: 2.2221906 (670)	total: 17.2s	remaining: 3m
698:	learn: 2.3064267	test: 2.2232545	best: 2.2221906 (670)	total: 17.3s	remaining: 3m
700:	learn: 2.3058289	test: 2.2232187	best: 2.2221906 (670)	total: 17.3s	remaining: 3m
702:	learn: 2.3053252	test: 2.2230821	best: 2.2221906 (670)	total: 17.4s	remaining: 3m
704:	learn: 2.3043279	test: 2.2225066	best: 2.2221906 (670)	total: 17.4s	remaining: 3m
706:	learn: 2.3033726	test: 2.2232222	best: 2.2221906 (670)	total: 17.5s	remaining: 3m
708:	learn: 2.3032396	test: 2.2231804	best: 2.2221906 (670)	total: 17.5s	remaining: 3m
710:	learn: 2.3024944	test: 2.2231441	best: 2.2221906 (670)	total: 17.6s	remaining: 3m
712:	learn: 2.3018448	test: 2.2235373	best: 2.2221906 (670)	total: 17.7s	remaining: 3m
714:	learn: 2.3010715	test: 2.2231115	best: 2.2221906 (670)	total: 17.7s	remaining: 3m
716:	learn: 2.3005151	test: 2.2229462	best: 2.2221906 (670)	total: 17.7s	remaining: 3m
718:	learn: 2.3004599	test: 2.2229654	best: 2.2221906 (670)	total: 17.8s	remaining: 3m
720:	learn: 2.3002841	test: 2.2230185	best: 2.2221906 (670)	total: 17.9s	remaining: 3m
722:	learn: 2.2994942	test: 2.2234410	best: 2.2221906 (670)	total: 17.9s	remaining: 3m
724:	learn: 2.2992171	test: 2.2234880	best: 2.2221906 (670)	total: 18s	remaining: 3m
726:	learn: 2.2975649	test: 2.2227927	best: 2.2221906 (670)	total: 18s	remaining: 3m
728:	learn: 2.2972412	test: 2.2227633	best: 2.2221906 (670)	total: 18.1s	remaining: 3m
730:	learn: 2.2966428	test: 2.2227935	best: 2.2221906 (670)	total: 18.1s	remaining: 3m
732:	learn: 2.2965005	test: 2.2227457	best: 2.2221906 (670)	total: 18.2s	remaining: 3m
734:	learn: 2.2962755	test: 2.2227216	best: 2.2221906 (670)	total: 18.2s	remaining: 3m
736:	learn: 2.2962054	test: 2.2226878	best: 2.2221906 (670)	total: 18.3s	remaining: 3m
738:	learn: 2.2957748	test: 2.2220962	best: 2.2220962 (738)	total: 18.3s	remaining: 3m
740:	learn: 2.2952375	test: 2.2221130	best: 2.2220009 (739)	total: 18.4s	remaining: 3m
742:	learn: 2.2947028	test: 2.2220896	best: 2.2220009 (739)	total: 18.4s	remaining: 3m
744:	learn: 2.2942120	test: 2.2216807	best: 2.2216807 (744)	total: 18.5s	remaining: 3m
746:	learn: 2.2937586	test: 2.2212436	best: 2.2212436 (746)	total: 18.6s	remaining: 3m
748:	learn: 2.2934249	test: 2.2217819	best: 2.2212436 (746)	total: 18.6s	remaining: 3m
750:	learn: 2.2924070	test: 2.2219252	best: 2.2212436 (746)	total: 18.7s	remaining: 3m
752:	learn: 2.2921535	test: 2.2218564	best: 2.2212436 (746)	total: 18.7s	remaining: 3m
754:	learn: 2.2906199	test: 2.2222739	best: 2.2212436 (746)	total: 18.8s	remaining: 3m
756:	learn: 2.2901001	test: 2.2220379	best: 2.2212436 (746)	total: 18.8s	remaining: 3m
758:	learn: 2.2895416	test: 2.2221754	best: 2.2212436 (746)	total: 18.9s	remaining: 3m
760:	learn: 2.2890682	test: 2.2219892	best: 2.2212436 (746)	total: 18.9s	remaining: 3m
762:	learn: 2.2886903	test: 2.2213809	best: 2.2212436 (746)	total: 19s	remaining: 3m
764:	learn: 2.2885090	test: 2.2212322	best: 2.2212322 (764)	total: 19s	remaining: 3m
766:	learn: 2.2880609	test: 2.2209609	best: 2.2209609 (766)	total: 19.1s	remaining: 3m
768:	learn: 2.2877820	test: 2.2209273	best: 2.2209273 (768)	total: 19.1s	remaining: 2m 59s
770:	learn: 2.2873451	test: 2.2202941	best: 2.2202941 (770)	total: 19.2s	remaining: 2m 59s
772:	learn: 2.2869336	test: 2.2200967	best: 2.2200967 (772)	total: 19.2s	remaining: 2m 59s
774:	learn: 2.2867636	test: 2.2200487	best: 2.2200487 (774)	total: 19.3s	remaining: 2m 59s
776:	learn: 2.2862212	test: 2.2201150	best: 2.2200487 (774)	total: 19.4s	remaining: 2m 59s
778:	learn: 2.2857480	test: 2.2203869	best: 2.2200487 (774)	total: 19.4s	remaining: 2m 59s
780:	learn: 2.2845883	test: 2.2200540	best: 2.2200487 (774)	total: 19.5s	remaining: 2m 59s
782:	learn: 2.2844364	test: 2.2200164	best: 2.2200164 (782)	total: 19.5s	remaining: 2m 59s
784:	learn: 2.2838194	test: 2.2191754	best: 2.2191754 (784)	total: 19.6s	remaining: 2m 59s
786:	learn: 2.2836804	test: 2.2190795	best: 2.2190795 (786)	total: 19.6s	remaining: 2m 59s
788:	learn: 2.2835685	test: 2.2190690	best: 2.2190690 (788)	total: 19.7s	remaining: 2m 59s
790:	learn: 2.2833225	test: 2.2191742	best: 2.2190690 (788)	total: 19.7s	remaining: 2m 59s
792:	learn: 2.2831117	test: 2.2191355	best: 2.2190690 (788)	total: 19.8s	remaining: 2m 59s
794:	learn: 2.2818314	test: 2.2191654	best: 2.2190690 (788)	total: 19.8s	remaining: 2m 59s
796:	learn: 2.2817461	test: 2.2191213	best: 2.2190690 (788)	total: 19.9s	remaining: 2m 59s
798:	learn: 2.2812011	test: 2.2192608	best: 2.2190690 (788)	total: 20s	remaining: 2m 59s
800:	learn: 2.2809759	test: 2.2199671	best: 2.2190690 (788)	total: 20s	remaining: 2m 59s
802:	learn: 2.2807560	test: 2.2197695	best: 2.2190690 (788)	total: 20.1s	remaining: 2m 59s
804:	learn: 2.2798686	test: 2.2201243	best: 2.2190690 (788)	total: 20.1s	remaining: 2m 59s
806:	learn: 2.2795261	test: 2.2199184	best: 2.2190690 (788)	total: 20.2s	remaining: 3m
808:	learn: 2.2788433	test: 2.2199213	best: 2.2190690 (788)	total: 20.3s	remaining: 3m
810:	learn: 2.2782419	test: 2.2201740	best: 2.2190690 (788)	total: 20.3s	remaining: 3m
812:	learn: 2.2780451	test: 2.2205473	best: 2.2190690 (788)	total: 20.4s	remaining: 3m
814:	learn: 2.2775670	test: 2.2203256	best: 2.2190690 (788)	total: 20.4s	remaining: 3m
816:	learn: 2.2770430	test: 2.2201592	best: 2.2190690 (788)	total: 20.5s	remaining: 3m
818:	learn: 2.2767140	test: 2.2201909	best: 2.2190690 (788)	total: 20.5s	remaining: 3m
820:	learn: 2.2763654	test: 2.2201087	best: 2.2190690 (788)	total: 20.6s	remaining: 3m
822:	learn: 2.2741967	test: 2.2199113	best: 2.2190690 (788)	total: 20.7s	remaining: 3m
824:	learn: 2.2735489	test: 2.2199742	best: 2.2190690 (788)	total: 20.7s	remaining: 3m
826:	learn: 2.2732001	test: 2.2199250	best: 2.2190690 (788)	total: 20.8s	remaining: 3m
828:	learn: 2.2730237	test: 2.2197313	best: 2.2190690 (788)	total: 20.8s	remaining: 3m
830:	learn: 2.2727594	test: 2.2195759	best: 2.2190690 (788)	total: 20.9s	remaining: 3m
832:	learn: 2.2720734	test: 2.2192904	best: 2.2190690 (788)	total: 20.9s	remaining: 3m
834:	learn: 2.2719712	test: 2.2192262	best: 2.2190690 (788)	total: 21s	remaining: 3m
836:	learn: 2.2716996	test: 2.2190936	best: 2.2190690 (788)	total: 21.1s	remaining: 3m
838:	learn: 2.2708498	test: 2.2189975	best: 2.2189975 (838)	total: 21.1s	remaining: 3m
840:	learn: 2.2699781	test: 2.2190985	best: 2.2189975 (838)	total: 21.2s	remaining: 3m
842:	learn: 2.2697159	test: 2.2191352	best: 2.2189975 (838)	total: 21.2s	remaining: 3m
844:	learn: 2.2693165	test: 2.2192252	best: 2.2189975 (838)	total: 21.3s	remaining: 3m
846:	learn: 2.2690266	test: 2.2200089	best: 2.2189975 (838)	total: 21.3s	remaining: 3m
848:	learn: 2.2686331	test: 2.2201037	best: 2.2189975 (838)	total: 21.4s	remaining: 3m
850:	learn: 2.2682934	test: 2.2198759	best: 2.2189975 (838)	total: 21.5s	remaining: 3m
852:	learn: 2.2678435	test: 2.2197168	best: 2.2189975 (838)	total: 21.5s	remaining: 3m
854:	learn: 2.2676706	test: 2.2197832	best: 2.2189975 (838)	total: 21.6s	remaining: 3m
856:	learn: 2.2664896	test: 2.2197282	best: 2.2189975 (838)	total: 21.6s	remaining: 3m
858:	learn: 2.2661257	test: 2.2195352	best: 2.2189975 (838)	total: 21.7s	remaining: 3m
860:	learn: 2.2657969	test: 2.2195761	best: 2.2189975 (838)	total: 21.7s	remaining: 3m
862:	learn: 2.2655271	test: 2.2195585	best: 2.2189975 (838)	total: 21.8s	remaining: 3m
864:	learn: 2.2650765	test: 2.2195380	best: 2.2189975 (838)	total: 21.8s	remaining: 3m
866:	learn: 2.2644169	test: 2.2202165	best: 2.2189975 (838)	total: 21.9s	remaining: 3m
868:	learn: 2.2640549	test: 2.2200690	best: 2.2189975 (838)	total: 21.9s	remaining: 3m
870:	learn: 2.2636307	test: 2.2199804	best: 2.2189975 (838)	total: 22s	remaining: 3m
872:	learn: 2.2629582	test: 2.2203338	best: 2.2189975 (838)	total: 22.1s	remaining: 3m
874:	learn: 2.2620986	test: 2.2199652	best: 2.2189975 (838)	total: 22.1s	remaining: 3m
876:	learn: 2.2616032	test: 2.2198208	best: 2.2189975 (838)	total: 22.2s	remaining: 3m
878:	learn: 2.2608440	test: 2.2192560	best: 2.2189975 (838)	total: 22.2s	remaining: 3m
880:	learn: 2.2602168	test: 2.2186179	best: 2.2186058 (879)	total: 22.3s	remaining: 2m 59s
882:	learn: 2.2599712	test: 2.2185440	best: 2.2185440 (882)	total: 22.3s	remaining: 2m 59s
884:	learn: 2.2596110	test: 2.2183374	best: 2.2183374 (884)	total: 22.4s	remaining: 2m 59s
886:	learn: 2.2593569	test: 2.2185893	best: 2.2183374 (884)	total: 22.4s	remaining: 2m 59s
888:	learn: 2.2587028	test: 2.2186021	best: 2.2183374 (884)	total: 22.5s	remaining: 2m 59s
890:	learn: 2.2584328	test: 2.2186269	best: 2.2183374 (884)	total: 22.5s	remaining: 2m 59s
892:	learn: 2.2580034	test: 2.2182333	best: 2.2182333 (892)	total: 22.6s	remaining: 2m 59s
894:	learn: 2.2575233	test: 2.2181023	best: 2.2181023 (894)	total: 22.6s	remaining: 2m 59s
896:	learn: 2.2566128	test: 2.2180333	best: 2.2180333 (896)	total: 22.7s	remaining: 2m 59s
898:	learn: 2.2560095	test: 2.2177901	best: 2.2177901 (898)	total: 22.7s	remaining: 2m 59s
900:	learn: 2.2557686	test: 2.2181215	best: 2.2177901 (898)	total: 22.8s	remaining: 2m 59s
902:	learn: 2.2554838	test: 2.2178679	best: 2.2177901 (898)	total: 22.9s	remaining: 2m 59s
904:	learn: 2.2550966	test: 2.2178360	best: 2.2177901 (898)	total: 22.9s	remaining: 2m 59s
906:	learn: 2.2549029	test: 2.2177175	best: 2.2177175 (906)	total: 23s	remaining: 2m 59s
908:	learn: 2.2538249	test: 2.2184431	best: 2.2174556 (907)	total: 23s	remaining: 2m 59s
910:	learn: 2.2534090	test: 2.2185360	best: 2.2174556 (907)	total: 23.1s	remaining: 2m 59s
912:	learn: 2.2527824	test: 2.2188215	best: 2.2174556 (907)	total: 23.1s	remaining: 2m 59s
914:	learn: 2.2526721	test: 2.2188653	best: 2.2174556 (907)	total: 23.2s	remaining: 2m 59s
916:	learn: 2.2521712	test: 2.2188990	best: 2.2174556 (907)	total: 23.2s	remaining: 2m 59s
918:	learn: 2.2518999	test: 2.2188077	best: 2.2174556 (907)	total: 23.3s	remaining: 2m 59s
920:	learn: 2.2517373	test: 2.2189996	best: 2.2174556 (907)	total: 23.3s	remaining: 2m 59s
922:	learn: 2.2513251	test: 2.2189956	best: 2.2174556 (907)	total: 23.4s	remaining: 2m 59s
924:	learn: 2.2511219	test: 2.2190917	best: 2.2174556 (907)	total: 23.4s	remaining: 2m 59s
926:	learn: 2.2491573	test: 2.2192934	best: 2.2174556 (907)	total: 23.5s	remaining: 2m 59s
928:	learn: 2.2477460	test: 2.2192026	best: 2.2174556 (907)	total: 23.5s	remaining: 2m 59s
930:	learn: 2.2473628	test: 2.2193628	best: 2.2174556 (907)	total: 23.6s	remaining: 2m 59s
932:	learn: 2.2469812	test: 2.2193582	best: 2.2174556 (907)	total: 23.6s	remaining: 2m 58s
934:	learn: 2.2466777	test: 2.2196726	best: 2.2174556 (907)	total: 23.7s	remaining: 2m 59s
936:	learn: 2.2463782	test: 2.2198578	best: 2.2174556 (907)	total: 23.8s	remaining: 2m 59s
938:	learn: 2.2458627	test: 2.2200745	best: 2.2174556 (907)	total: 23.8s	remaining: 2m 58s
940:	learn: 2.2455115	test: 2.2196145	best: 2.2174556 (907)	total: 23.8s	remaining: 2m 58s
942:	learn: 2.2453670	test: 2.2195766	best: 2.2174556 (907)	total: 23.9s	remaining: 2m 58s
944:	learn: 2.2447818	test: 2.2195059	best: 2.2174556 (907)	total: 24s	remaining: 2m 58s
946:	learn: 2.2439783	test: 2.2193635	best: 2.2174556 (907)	total: 24s	remaining: 2m 58s
948:	learn: 2.2430092	test: 2.2197625	best: 2.2174556 (907)	total: 24.1s	remaining: 2m 58s
950:	learn: 2.2420819	test: 2.2201372	best: 2.2174556 (907)	total: 24.1s	remaining: 2m 58s
952:	learn: 2.2417629	test: 2.2200756	best: 2.2174556 (907)	total: 24.2s	remaining: 2m 58s
954:	learn: 2.2415015	test: 2.2200298	best: 2.2174556 (907)	total: 24.2s	remaining: 2m 58s
956:	learn: 2.2414172	test: 2.2200297	best: 2.2174556 (907)	total: 24.3s	remaining: 2m 58s
958:	learn: 2.2412057	test: 2.2198446	best: 2.2174556 (907)	total: 24.3s	remaining: 2m 58s
960:	learn: 2.2403695	test: 2.2197400	best: 2.2174556 (907)	total: 24.4s	remaining: 2m 58s
962:	learn: 2.2398798	test: 2.2193825	best: 2.2174556 (907)	total: 24.4s	remaining: 2m 58s
964:	learn: 2.2394566	test: 2.2193915	best: 2.2174556 (907)	total: 24.5s	remaining: 2m 58s
966:	learn: 2.2388110	test: 2.2193524	best: 2.2174556 (907)	total: 24.5s	remaining: 2m 58s
968:	learn: 2.2377158	test: 2.2195492	best: 2.2174556 (907)	total: 24.6s	remaining: 2m 58s
970:	learn: 2.2376023	test: 2.2195593	best: 2.2174556 (907)	total: 24.6s	remaining: 2m 58s
972:	learn: 2.2368766	test: 2.2194428	best: 2.2174556 (907)	total: 24.7s	remaining: 2m 58s
974:	learn: 2.2365998	test: 2.2193512	best: 2.2174556 (907)	total: 24.8s	remaining: 2m 58s
976:	learn: 2.2363132	test: 2.2192146	best: 2.2174556 (907)	total: 24.8s	remaining: 2m 58s
978:	learn: 2.2355565	test: 2.2189354	best: 2.2174556 (907)	total: 24.9s	remaining: 2m 58s
980:	learn: 2.2352580	test: 2.2190626	best: 2.2174556 (907)	total: 24.9s	remaining: 2m 58s
982:	learn: 2.2343376	test: 2.2188519	best: 2.2174556 (907)	total: 25s	remaining: 2m 58s
984:	learn: 2.2340713	test: 2.2185924	best: 2.2174556 (907)	total: 25s	remaining: 2m 58s
986:	learn: 2.2334469	test: 2.2185373	best: 2.2174556 (907)	total: 25.1s	remaining: 2m 58s
988:	learn: 2.2332748	test: 2.2184835	best: 2.2174556 (907)	total: 25.1s	remaining: 2m 57s
990:	learn: 2.2331056	test: 2.2186414	best: 2.2174556 (907)	total: 25.2s	remaining: 2m 57s
992:	learn: 2.2329966	test: 2.2185816	best: 2.2174556 (907)	total: 25.2s	remaining: 2m 58s
994:	learn: 2.2328323	test: 2.2188030	best: 2.2174556 (907)	total: 25.3s	remaining: 2m 57s
996:	learn: 2.2325894	test: 2.2188836	best: 2.2174556 (907)	total: 25.3s	remaining: 2m 57s
998:	learn: 2.2319320	test: 2.2186955	best: 2.2174556 (907)	total: 25.4s	remaining: 2m 57s
1000:	learn: 2.2317131	test: 2.2190699	best: 2.2174556 (907)	total: 25.4s	remaining: 2m 57s
1002:	learn: 2.2316175	test: 2.2190138	best: 2.2174556 (907)	total: 25.5s	remaining: 2m 57s
1004:	learn: 2.2307631	test: 2.2190968	best: 2.2174556 (907)	total: 25.6s	remaining: 2m 57s
1006:	learn: 2.2305590	test: 2.2191176	best: 2.2174556 (907)	total: 25.6s	remaining: 2m 57s
1008:	learn: 2.2302906	test: 2.2191812	best: 2.2174556 (907)	total: 25.7s	remaining: 2m 57s
1010:	learn: 2.2298779	test: 2.2190725	best: 2.2174556 (907)	total: 25.7s	remaining: 2m 57s
1012:	learn: 2.2296793	test: 2.2190251	best: 2.2174556 (907)	total: 25.8s	remaining: 2m 57s
1014:	learn: 2.2295254	test: 2.2190368	best: 2.2174556 (907)	total: 25.8s	remaining: 2m 57s
1016:	learn: 2.2292310	test: 2.2189598	best: 2.2174556 (907)	total: 25.9s	remaining: 2m 57s
1018:	learn: 2.2288616	test: 2.2193086	best: 2.2174556 (907)	total: 25.9s	remaining: 2m 57s
1020:	learn: 2.2280078	test: 2.2186041	best: 2.2174556 (907)	total: 26s	remaining: 2m 57s
1022:	learn: 2.2275760	test: 2.2186056	best: 2.2174556 (907)	total: 26.1s	remaining: 2m 57s
1024:	learn: 2.2274462	test: 2.2185185	best: 2.2174556 (907)	total: 26.1s	remaining: 2m 57s
1026:	learn: 2.2271375	test: 2.2188536	best: 2.2174556 (907)	total: 26.2s	remaining: 2m 57s
1028:	learn: 2.2267960	test: 2.2190270	best: 2.2174556 (907)	total: 26.2s	remaining: 2m 57s
1030:	learn: 2.2265973	test: 2.2191391	best: 2.2174556 (907)	total: 26.3s	remaining: 2m 57s
1032:	learn: 2.2262554	test: 2.2191775	best: 2.2174556 (907)	total: 26.3s	remaining: 2m 57s
1034:	learn: 2.2259336	test: 2.2192495	best: 2.2174556 (907)	total: 26.4s	remaining: 2m 57s
1036:	learn: 2.2252424	test: 2.2194467	best: 2.2174556 (907)	total: 26.4s	remaining: 2m 57s
1038:	learn: 2.2245064	test: 2.2194604	best: 2.2174556 (907)	total: 26.5s	remaining: 2m 57s
1040:	learn: 2.2239122	test: 2.2195951	best: 2.2174556 (907)	total: 26.5s	remaining: 2m 57s
1042:	learn: 2.2237706	test: 2.2195879	best: 2.2174556 (907)	total: 26.6s	remaining: 2m 57s
1044:	learn: 2.2237110	test: 2.2197335	best: 2.2174556 (907)	total: 26.6s	remaining: 2m 57s
1046:	learn: 2.2235191	test: 2.2197070	best: 2.2174556 (907)	total: 26.7s	remaining: 2m 57s
1048:	learn: 2.2231054	test: 2.2190205	best: 2.2174556 (907)	total: 26.7s	remaining: 2m 57s
1050:	learn: 2.2229924	test: 2.2190118	best: 2.2174556 (907)	total: 26.8s	remaining: 2m 57s
1052:	learn: 2.2229392	test: 2.2189633	best: 2.2174556 (907)	total: 26.9s	remaining: 2m 57s
1054:	learn: 2.2226974	test: 2.2187545	best: 2.2174556 (907)	total: 26.9s	remaining: 2m 57s
1056:	learn: 2.2224108	test: 2.2184112	best: 2.2174556 (907)	total: 26.9s	remaining: 2m 57s
1058:	learn: 2.2221106	test: 2.2183771	best: 2.2174556 (907)	total: 27s	remaining: 2m 56s
1060:	learn: 2.2215340	test: 2.2180077	best: 2.2174556 (907)	total: 27s	remaining: 2m 56s
1062:	learn: 2.2212126	test: 2.2179365	best: 2.2174556 (907)	total: 27.1s	remaining: 2m 56s
1064:	learn: 2.2210098	test: 2.2179558	best: 2.2174556 (907)	total: 27.1s	remaining: 2m 56s
1066:	learn: 2.2199485	test: 2.2179359	best: 2.2174556 (907)	total: 27.2s	remaining: 2m 56s
1068:	learn: 2.2195564	test: 2.2180431	best: 2.2174556 (907)	total: 27.3s	remaining: 2m 56s
1070:	learn: 2.2190934	test: 2.2177694	best: 2.2174556 (907)	total: 27.3s	remaining: 2m 56s
1072:	learn: 2.2186468	test: 2.2179076	best: 2.2174556 (907)	total: 27.4s	remaining: 2m 56s
1074:	learn: 2.2182211	test: 2.2176949	best: 2.2174556 (907)	total: 27.4s	remaining: 2m 56s
1076:	learn: 2.2179032	test: 2.2177406	best: 2.2174556 (907)	total: 27.5s	remaining: 2m 56s
1078:	learn: 2.2174132	test: 2.2176597	best: 2.2174556 (907)	total: 27.5s	remaining: 2m 56s
1080:	learn: 2.2167837	test: 2.2173657	best: 2.2173657 (1080)	total: 27.6s	remaining: 2m 56s
1082:	learn: 2.2163891	test: 2.2171394	best: 2.2171394 (1082)	total: 27.6s	remaining: 2m 56s
1084:	learn: 2.2162372	test: 2.2170731	best: 2.2170731 (1084)	total: 27.7s	remaining: 2m 56s
1086:	learn: 2.2158791	test: 2.2168933	best: 2.2168140 (1085)	total: 27.7s	remaining: 2m 56s
1088:	learn: 2.2156258	test: 2.2168521	best: 2.2168140 (1085)	total: 27.8s	remaining: 2m 56s
1090:	learn: 2.2154219	test: 2.2168062	best: 2.2167815 (1089)	total: 27.9s	remaining: 2m 56s
1092:	learn: 2.2150231	test: 2.2165408	best: 2.2165408 (1092)	total: 27.9s	remaining: 2m 56s
1094:	learn: 2.2144391	test: 2.2167062	best: 2.2165345 (1093)	total: 28s	remaining: 2m 56s
1096:	learn: 2.2136777	test: 2.2172149	best: 2.2165345 (1093)	total: 28s	remaining: 2m 56s
1098:	learn: 2.2134472	test: 2.2172334	best: 2.2165345 (1093)	total: 28.1s	remaining: 2m 56s
1100:	learn: 2.2132648	test: 2.2172493	best: 2.2165345 (1093)	total: 28.1s	remaining: 2m 56s
1102:	learn: 2.2130866	test: 2.2172983	best: 2.2165345 (1093)	total: 28.1s	remaining: 2m 55s
1104:	learn: 2.2129264	test: 2.2173148	best: 2.2165345 (1093)	total: 28.2s	remaining: 2m 55s
1106:	learn: 2.2127101	test: 2.2171014	best: 2.2165345 (1093)	total: 28.2s	remaining: 2m 55s
1108:	learn: 2.2121352	test: 2.2168953	best: 2.2165345 (1093)	total: 28.3s	remaining: 2m 55s
1110:	learn: 2.2116945	test: 2.2166934	best: 2.2165345 (1093)	total: 28.3s	remaining: 2m 55s
1112:	learn: 2.2113752	test: 2.2167419	best: 2.2165345 (1093)	total: 28.4s	remaining: 2m 55s
1114:	learn: 2.2106079	test: 2.2166774	best: 2.2165345 (1093)	total: 28.4s	remaining: 2m 55s
1116:	learn: 2.2102240	test: 2.2167137	best: 2.2165345 (1093)	total: 28.5s	remaining: 2m 55s
1118:	learn: 2.2101572	test: 2.2165264	best: 2.2165264 (1118)	total: 28.6s	remaining: 2m 55s
1120:	learn: 2.2096534	test: 2.2164338	best: 2.2164338 (1120)	total: 28.6s	remaining: 2m 55s
1122:	learn: 2.2090997	test: 2.2168265	best: 2.2164338 (1120)	total: 28.7s	remaining: 2m 55s
1124:	learn: 2.2088620	test: 2.2167691	best: 2.2164338 (1120)	total: 28.7s	remaining: 2m 55s
1126:	learn: 2.2082409	test: 2.2167415	best: 2.2164338 (1120)	total: 28.8s	remaining: 2m 55s
1128:	learn: 2.2077099	test: 2.2169336	best: 2.2164338 (1120)	total: 28.8s	remaining: 2m 55s
1130:	learn: 2.2075815	test: 2.2169188	best: 2.2164338 (1120)	total: 28.9s	remaining: 2m 55s
1132:	learn: 2.2072464	test: 2.2168126	best: 2.2164338 (1120)	total: 28.9s	remaining: 2m 55s
1134:	learn: 2.2067668	test: 2.2166643	best: 2.2164338 (1120)	total: 29s	remaining: 2m 55s
1136:	learn: 2.2063560	test: 2.2165839	best: 2.2164338 (1120)	total: 29s	remaining: 2m 55s
1138:	learn: 2.2047307	test: 2.2165625	best: 2.2164338 (1120)	total: 29.1s	remaining: 2m 55s
1140:	learn: 2.2046214	test: 2.2168822	best: 2.2164338 (1120)	total: 29.1s	remaining: 2m 55s
1142:	learn: 2.2044374	test: 2.2167397	best: 2.2164338 (1120)	total: 29.2s	remaining: 2m 55s
1144:	learn: 2.2037902	test: 2.2167352	best: 2.2164338 (1120)	total: 29.3s	remaining: 2m 55s
1146:	learn: 2.2036928	test: 2.2166029	best: 2.2164338 (1120)	total: 29.3s	remaining: 2m 55s
1148:	learn: 2.2034903	test: 2.2165247	best: 2.2164338 (1120)	total: 29.4s	remaining: 2m 55s
1150:	learn: 2.2032982	test: 2.2166307	best: 2.2164338 (1120)	total: 29.4s	remaining: 2m 55s
1152:	learn: 2.2027550	test: 2.2165051	best: 2.2164338 (1120)	total: 29.5s	remaining: 2m 54s
1154:	learn: 2.2026526	test: 2.2163877	best: 2.2163877 (1154)	total: 29.5s	remaining: 2m 54s
1156:	learn: 2.2022509	test: 2.2165047	best: 2.2163877 (1154)	total: 29.6s	remaining: 2m 54s
1158:	learn: 2.2017897	test: 2.2161945	best: 2.2161945 (1158)	total: 29.6s	remaining: 2m 54s
1160:	learn: 2.2015242	test: 2.2160451	best: 2.2160451 (1160)	total: 29.7s	remaining: 2m 54s
1162:	learn: 2.2014273	test: 2.2160372	best: 2.2160357 (1161)	total: 29.7s	remaining: 2m 54s
1164:	learn: 2.2010230	test: 2.2159479	best: 2.2159417 (1163)	total: 29.8s	remaining: 2m 54s
1166:	learn: 2.2006332	test: 2.2158865	best: 2.2158865 (1166)	total: 29.9s	remaining: 2m 54s
1168:	learn: 2.2002941	test: 2.2159822	best: 2.2158865 (1166)	total: 29.9s	remaining: 2m 54s
1170:	learn: 2.1999709	test: 2.2160408	best: 2.2158865 (1166)	total: 30s	remaining: 2m 54s
1172:	learn: 2.1996763	test: 2.2162131	best: 2.2158865 (1166)	total: 30s	remaining: 2m 54s
1174:	learn: 2.1990244	test: 2.2162024	best: 2.2158865 (1166)	total: 30.1s	remaining: 2m 54s
1176:	learn: 2.1985585	test: 2.2162167	best: 2.2158865 (1166)	total: 30.1s	remaining: 2m 54s
1178:	learn: 2.1982973	test: 2.2160485	best: 2.2158865 (1166)	total: 30.2s	remaining: 2m 54s
1180:	learn: 2.1977195	test: 2.2160757	best: 2.2158865 (1166)	total: 30.3s	remaining: 2m 54s
1182:	learn: 2.1976348	test: 2.2160928	best: 2.2158865 (1166)	total: 30.3s	remaining: 2m 54s
1184:	learn: 2.1972186	test: 2.2161717	best: 2.2158865 (1166)	total: 30.4s	remaining: 2m 54s
1186:	learn: 2.1970354	test: 2.2162968	best: 2.2158865 (1166)	total: 30.4s	remaining: 2m 54s
1188:	learn: 2.1969012	test: 2.2163186	best: 2.2158865 (1166)	total: 30.5s	remaining: 2m 54s
1190:	learn: 2.1966969	test: 2.2161841	best: 2.2158865 (1166)	total: 30.5s	remaining: 2m 54s
1192:	learn: 2.1964679	test: 2.2161367	best: 2.2158865 (1166)	total: 30.6s	remaining: 2m 54s
1194:	learn: 2.1963225	test: 2.2163774	best: 2.2158865 (1166)	total: 30.6s	remaining: 2m 54s
1196:	learn: 2.1960014	test: 2.2164111	best: 2.2158865 (1166)	total: 30.7s	remaining: 2m 54s
1198:	learn: 2.1957729	test: 2.2164832	best: 2.2158865 (1166)	total: 30.7s	remaining: 2m 54s
1200:	learn: 2.1957289	test: 2.2165542	best: 2.2158865 (1166)	total: 30.8s	remaining: 2m 54s
1202:	learn: 2.1954654	test: 2.2165444	best: 2.2158865 (1166)	total: 30.8s	remaining: 2m 54s
1204:	learn: 2.1945697	test: 2.2165592	best: 2.2158865 (1166)	total: 30.9s	remaining: 2m 54s
1206:	learn: 2.1933274	test: 2.2164921	best: 2.2158865 (1166)	total: 31s	remaining: 2m 54s
1208:	learn: 2.1928639	test: 2.2166501	best: 2.2158865 (1166)	total: 31s	remaining: 2m 54s
1210:	learn: 2.1923214	test: 2.2164261	best: 2.2158865 (1166)	total: 31.1s	remaining: 2m 54s
1212:	learn: 2.1920347	test: 2.2164011	best: 2.2158865 (1166)	total: 31.2s	remaining: 2m 54s
1214:	learn: 2.1915268	test: 2.2163060	best: 2.2158865 (1166)	total: 31.3s	remaining: 2m 54s
1216:	learn: 2.1913436	test: 2.2164805	best: 2.2158865 (1166)	total: 31.3s	remaining: 2m 54s
1218:	learn: 2.1910774	test: 2.2164496	best: 2.2158865 (1166)	total: 31.4s	remaining: 2m 54s
1220:	learn: 2.1898808	test: 2.2162997	best: 2.2158865 (1166)	total: 31.5s	remaining: 2m 54s
1222:	learn: 2.1897802	test: 2.2162379	best: 2.2158865 (1166)	total: 31.5s	remaining: 2m 54s
1224:	learn: 2.1895689	test: 2.2161299	best: 2.2158865 (1166)	total: 31.6s	remaining: 2m 54s
1226:	learn: 2.1888934	test: 2.2159456	best: 2.2158865 (1166)	total: 31.6s	remaining: 2m 54s
1228:	learn: 2.1883647	test: 2.2158721	best: 2.2158721 (1228)	total: 31.7s	remaining: 2m 54s
1230:	learn: 2.1880993	test: 2.2160377	best: 2.2158721 (1228)	total: 31.7s	remaining: 2m 54s
1232:	learn: 2.1875259	test: 2.2161182	best: 2.2158721 (1228)	total: 31.8s	remaining: 2m 54s
1234:	learn: 2.1874146	test: 2.2159581	best: 2.2158721 (1228)	total: 31.8s	remaining: 2m 54s
1236:	learn: 2.1871278	test: 2.2162225	best: 2.2158721 (1228)	total: 31.9s	remaining: 2m 54s
1238:	learn: 2.1867211	test: 2.2162460	best: 2.2158721 (1228)	total: 32s	remaining: 2m 54s
1240:	learn: 2.1864597	test: 2.2163521	best: 2.2158721 (1228)	total: 32s	remaining: 2m 54s
1242:	learn: 2.1863098	test: 2.2164510	best: 2.2158721 (1228)	total: 32.1s	remaining: 2m 54s
1244:	learn: 2.1860420	test: 2.2165468	best: 2.2158721 (1228)	total: 32.1s	remaining: 2m 54s
1246:	learn: 2.1856842	test: 2.2164622	best: 2.2158721 (1228)	total: 32.2s	remaining: 2m 54s
1248:	learn: 2.1850765	test: 2.2162931	best: 2.2158721 (1228)	total: 32.2s	remaining: 2m 54s
1250:	learn: 2.1849571	test: 2.2163012	best: 2.2158721 (1228)	total: 32.3s	remaining: 2m 54s
1252:	learn: 2.1845862	test: 2.2169046	best: 2.2158721 (1228)	total: 32.3s	remaining: 2m 54s
1254:	learn: 2.1834885	test: 2.2166156	best: 2.2158721 (1228)	total: 32.4s	remaining: 2m 54s
1256:	learn: 2.1832343	test: 2.2167414	best: 2.2158721 (1228)	total: 32.4s	remaining: 2m 54s
1258:	learn: 2.1830993	test: 2.2166337	best: 2.2158721 (1228)	total: 32.5s	remaining: 2m 54s
1260:	learn: 2.1815204	test: 2.2162470	best: 2.2158721 (1228)	total: 32.6s	remaining: 2m 54s
1262:	learn: 2.1798983	test: 2.2161090	best: 2.2158721 (1228)	total: 32.6s	remaining: 2m 53s
1264:	learn: 2.1795402	test: 2.2160459	best: 2.2158721 (1228)	total: 32.7s	remaining: 2m 53s
1266:	learn: 2.1792816	test: 2.2159929	best: 2.2158721 (1228)	total: 32.7s	remaining: 2m 53s
1268:	learn: 2.1777121	test: 2.2159904	best: 2.2158721 (1228)	total: 32.8s	remaining: 2m 53s
1270:	learn: 2.1771780	test: 2.2159027	best: 2.2158721 (1228)	total: 32.9s	remaining: 2m 53s
1272:	learn: 2.1765393	test: 2.2160619	best: 2.2158721 (1228)	total: 32.9s	remaining: 2m 53s
1274:	learn: 2.1761694	test: 2.2154081	best: 2.2154081 (1274)	total: 33s	remaining: 2m 53s
1276:	learn: 2.1758557	test: 2.2154434	best: 2.2154081 (1274)	total: 33s	remaining: 2m 53s
1278:	learn: 2.1757168	test: 2.2155513	best: 2.2154081 (1274)	total: 33.1s	remaining: 2m 53s
1280:	learn: 2.1754587	test: 2.2156020	best: 2.2154081 (1274)	total: 33.1s	remaining: 2m 53s
1282:	learn: 2.1750310	test: 2.2155528	best: 2.2154081 (1274)	total: 33.2s	remaining: 2m 53s
1284:	learn: 2.1744925	test: 2.2155456	best: 2.2154081 (1274)	total: 33.2s	remaining: 2m 53s
1286:	learn: 2.1741291	test: 2.2155210	best: 2.2154081 (1274)	total: 33.3s	remaining: 2m 53s
1288:	learn: 2.1738782	test: 2.2156223	best: 2.2154081 (1274)	total: 33.4s	remaining: 2m 53s
1290:	learn: 2.1735475	test: 2.2155653	best: 2.2154081 (1274)	total: 33.4s	remaining: 2m 53s
1292:	learn: 2.1728506	test: 2.2155906	best: 2.2154081 (1274)	total: 33.5s	remaining: 2m 53s
1294:	learn: 2.1724193	test: 2.2154548	best: 2.2154081 (1274)	total: 33.5s	remaining: 2m 53s
1296:	learn: 2.1720828	test: 2.2154256	best: 2.2154081 (1274)	total: 33.6s	remaining: 2m 53s
1298:	learn: 2.1714982	test: 2.2153163	best: 2.2153163 (1298)	total: 33.6s	remaining: 2m 53s
1300:	learn: 2.1712063	test: 2.2150680	best: 2.2150680 (1300)	total: 33.7s	remaining: 2m 53s
1302:	learn: 2.1708918	test: 2.2150702	best: 2.2150680 (1300)	total: 33.8s	remaining: 2m 53s
1304:	learn: 2.1707407	test: 2.2151567	best: 2.2150680 (1300)	total: 33.9s	remaining: 2m 53s
1306:	learn: 2.1700180	test: 2.2152215	best: 2.2150680 (1300)	total: 33.9s	remaining: 2m 53s
1308:	learn: 2.1696006	test: 2.2155589	best: 2.2150680 (1300)	total: 34s	remaining: 2m 53s
1310:	learn: 2.1694619	test: 2.2154991	best: 2.2150680 (1300)	total: 34.1s	remaining: 2m 53s
1312:	learn: 2.1688629	test: 2.2155814	best: 2.2150680 (1300)	total: 34.1s	remaining: 2m 53s
1314:	learn: 2.1679081	test: 2.2155985	best: 2.2150680 (1300)	total: 34.2s	remaining: 2m 53s
1316:	learn: 2.1677945	test: 2.2155936	best: 2.2150680 (1300)	total: 34.2s	remaining: 2m 53s
1318:	learn: 2.1674106	test: 2.2155425	best: 2.2150680 (1300)	total: 34.3s	remaining: 2m 53s
1320:	learn: 2.1670103	test: 2.2155261	best: 2.2150680 (1300)	total: 34.4s	remaining: 2m 53s
1322:	learn: 2.1668509	test: 2.2154311	best: 2.2150680 (1300)	total: 34.4s	remaining: 2m 53s
1324:	learn: 2.1667049	test: 2.2154541	best: 2.2150680 (1300)	total: 34.5s	remaining: 2m 53s
1326:	learn: 2.1664162	test: 2.2154824	best: 2.2150680 (1300)	total: 34.5s	remaining: 2m 53s
1328:	learn: 2.1663955	test: 2.2154844	best: 2.2150680 (1300)	total: 34.6s	remaining: 2m 53s
1330:	learn: 2.1662765	test: 2.2154557	best: 2.2150680 (1300)	total: 34.7s	remaining: 2m 53s
1332:	learn: 2.1655406	test: 2.2154613	best: 2.2150680 (1300)	total: 34.7s	remaining: 2m 53s
1334:	learn: 2.1651266	test: 2.2155222	best: 2.2150680 (1300)	total: 34.8s	remaining: 2m 53s
1336:	learn: 2.1649065	test: 2.2155329	best: 2.2150680 (1300)	total: 34.8s	remaining: 2m 53s
1338:	learn: 2.1646466	test: 2.2156461	best: 2.2150680 (1300)	total: 34.9s	remaining: 2m 53s
1340:	learn: 2.1644706	test: 2.2154404	best: 2.2150680 (1300)	total: 34.9s	remaining: 2m 53s
1342:	learn: 2.1641397	test: 2.2152122	best: 2.2150680 (1300)	total: 35s	remaining: 2m 53s
1344:	learn: 2.1639661	test: 2.2152899	best: 2.2150680 (1300)	total: 35s	remaining: 2m 53s
1346:	learn: 2.1634560	test: 2.2152570	best: 2.2150680 (1300)	total: 35.1s	remaining: 2m 53s
1348:	learn: 2.1626932	test: 2.2154320	best: 2.2150680 (1300)	total: 35.2s	remaining: 2m 53s
1350:	learn: 2.1624896	test: 2.2154363	best: 2.2150680 (1300)	total: 35.2s	remaining: 2m 53s
1352:	learn: 2.1619440	test: 2.2154320	best: 2.2150680 (1300)	total: 35.3s	remaining: 2m 53s
1354:	learn: 2.1612894	test: 2.2153991	best: 2.2150680 (1300)	total: 35.4s	remaining: 2m 53s
1356:	learn: 2.1605686	test: 2.2153130	best: 2.2150680 (1300)	total: 35.4s	remaining: 2m 53s
1358:	learn: 2.1603603	test: 2.2152774	best: 2.2150680 (1300)	total: 35.5s	remaining: 2m 53s
1360:	learn: 2.1602425	test: 2.2152406	best: 2.2150680 (1300)	total: 35.5s	remaining: 2m 53s
1362:	learn: 2.1600483	test: 2.2152833	best: 2.2150680 (1300)	total: 35.6s	remaining: 2m 53s
1364:	learn: 2.1590095	test: 2.2148076	best: 2.2148076 (1364)	total: 35.6s	remaining: 2m 53s
1366:	learn: 2.1587636	test: 2.2146869	best: 2.2146869 (1366)	total: 35.7s	remaining: 2m 53s
1368:	learn: 2.1583554	test: 2.2146613	best: 2.2146613 (1368)	total: 35.7s	remaining: 2m 53s
1370:	learn: 2.1577184	test: 2.2148903	best: 2.2146520 (1369)	total: 35.8s	remaining: 2m 53s
1372:	learn: 2.1572900	test: 2.2146500	best: 2.2146353 (1371)	total: 35.9s	remaining: 2m 53s
1374:	learn: 2.1568505	test: 2.2146722	best: 2.2146353 (1371)	total: 35.9s	remaining: 2m 53s
1376:	learn: 2.1567883	test: 2.2146499	best: 2.2146353 (1371)	total: 36s	remaining: 2m 53s
1378:	learn: 2.1562627	test: 2.2146026	best: 2.2146026 (1378)	total: 36.1s	remaining: 2m 53s
1380:	learn: 2.1560488	test: 2.2145989	best: 2.2145964 (1379)	total: 36.1s	remaining: 2m 53s
1382:	learn: 2.1557650	test: 2.2142669	best: 2.2142325 (1381)	total: 36.2s	remaining: 2m 53s
1384:	learn: 2.1555889	test: 2.2141116	best: 2.2141011 (1383)	total: 36.2s	remaining: 2m 53s
1386:	learn: 2.1548131	test: 2.2141024	best: 2.2141011 (1383)	total: 36.3s	remaining: 2m 53s
1388:	learn: 2.1545963	test: 2.2141243	best: 2.2140542 (1387)	total: 36.4s	remaining: 2m 53s
1390:	learn: 2.1542545	test: 2.2139894	best: 2.2139894 (1390)	total: 36.4s	remaining: 2m 53s
1392:	learn: 2.1542125	test: 2.2139824	best: 2.2139824 (1392)	total: 36.5s	remaining: 2m 53s
1394:	learn: 2.1540506	test: 2.2139212	best: 2.2139212 (1394)	total: 36.6s	remaining: 2m 53s
1396:	learn: 2.1538616	test: 2.2138638	best: 2.2138638 (1396)	total: 36.6s	remaining: 2m 53s
1398:	learn: 2.1534918	test: 2.2138666	best: 2.2138638 (1396)	total: 36.7s	remaining: 2m 52s
1400:	learn: 2.1533299	test: 2.2138633	best: 2.2138417 (1399)	total: 36.7s	remaining: 2m 53s
1402:	learn: 2.1530857	test: 2.2138854	best: 2.2138417 (1399)	total: 36.8s	remaining: 2m 52s
1404:	learn: 2.1524366	test: 2.2136832	best: 2.2136832 (1404)	total: 36.8s	remaining: 2m 52s
1406:	learn: 2.1521308	test: 2.2138113	best: 2.2136832 (1404)	total: 36.9s	remaining: 2m 52s
1408:	learn: 2.1518047	test: 2.2138352	best: 2.2136832 (1404)	total: 37s	remaining: 2m 52s
1410:	learn: 2.1516857	test: 2.2138430	best: 2.2136832 (1404)	total: 37s	remaining: 2m 52s
1412:	learn: 2.1515109	test: 2.2138420	best: 2.2136832 (1404)	total: 37.1s	remaining: 2m 52s
1414:	learn: 2.1511715	test: 2.2140250	best: 2.2136832 (1404)	total: 37.1s	remaining: 2m 52s
1416:	learn: 2.1509648	test: 2.2143315	best: 2.2136832 (1404)	total: 37.2s	remaining: 2m 52s
1418:	learn: 2.1507283	test: 2.2143551	best: 2.2136832 (1404)	total: 37.3s	remaining: 2m 52s
1420:	learn: 2.1505518	test: 2.2141748	best: 2.2136832 (1404)	total: 37.3s	remaining: 2m 52s
1422:	learn: 2.1504037	test: 2.2140765	best: 2.2136832 (1404)	total: 37.4s	remaining: 2m 52s
1424:	learn: 2.1501504	test: 2.2139575	best: 2.2136832 (1404)	total: 37.4s	remaining: 2m 52s
1426:	learn: 2.1499557	test: 2.2137249	best: 2.2136832 (1404)	total: 37.5s	remaining: 2m 52s
1428:	learn: 2.1498279	test: 2.2137253	best: 2.2136832 (1404)	total: 37.5s	remaining: 2m 52s
1430:	learn: 2.1495933	test: 2.2136441	best: 2.2136210 (1429)	total: 37.6s	remaining: 2m 52s
1432:	learn: 2.1492493	test: 2.2133897	best: 2.2133897 (1432)	total: 37.6s	remaining: 2m 52s
1434:	learn: 2.1488953	test: 2.2134321	best: 2.2133807 (1433)	total: 37.7s	remaining: 2m 52s
1436:	learn: 2.1488034	test: 2.2135901	best: 2.2133807 (1433)	total: 37.7s	remaining: 2m 52s
1438:	learn: 2.1485643	test: 2.2134035	best: 2.2133807 (1433)	total: 37.8s	remaining: 2m 52s
1440:	learn: 2.1481140	test: 2.2137194	best: 2.2133807 (1433)	total: 37.9s	remaining: 2m 52s
1442:	learn: 2.1477287	test: 2.2136573	best: 2.2133807 (1433)	total: 37.9s	remaining: 2m 52s
1444:	learn: 2.1475301	test: 2.2135914	best: 2.2133807 (1433)	total: 38s	remaining: 2m 52s
1446:	learn: 2.1470399	test: 2.2137284	best: 2.2133807 (1433)	total: 38s	remaining: 2m 52s
1448:	learn: 2.1468855	test: 2.2137862	best: 2.2133807 (1433)	total: 38.1s	remaining: 2m 52s
1450:	learn: 2.1464558	test: 2.2137170	best: 2.2133807 (1433)	total: 38.1s	remaining: 2m 52s
1452:	learn: 2.1461696	test: 2.2137374	best: 2.2133807 (1433)	total: 38.2s	remaining: 2m 52s
1454:	learn: 2.1460966	test: 2.2137494	best: 2.2133807 (1433)	total: 38.2s	remaining: 2m 52s
1456:	learn: 2.1459067	test: 2.2137330	best: 2.2133807 (1433)	total: 38.3s	remaining: 2m 52s
1458:	learn: 2.1457531	test: 2.2137398	best: 2.2133807 (1433)	total: 38.4s	remaining: 2m 51s
1460:	learn: 2.1455071	test: 2.2139031	best: 2.2133807 (1433)	total: 38.4s	remaining: 2m 51s
1462:	learn: 2.1449349	test: 2.2138709	best: 2.2133807 (1433)	total: 38.5s	remaining: 2m 51s
1464:	learn: 2.1448021	test: 2.2139263	best: 2.2133807 (1433)	total: 38.5s	remaining: 2m 51s
1466:	learn: 2.1445499	test: 2.2139552	best: 2.2133807 (1433)	total: 38.6s	remaining: 2m 51s
1468:	learn: 2.1443240	test: 2.2139981	best: 2.2133807 (1433)	total: 38.6s	remaining: 2m 51s
1470:	learn: 2.1439193	test: 2.2138251	best: 2.2133807 (1433)	total: 38.7s	remaining: 2m 51s
1472:	learn: 2.1437887	test: 2.2138966	best: 2.2133807 (1433)	total: 38.7s	remaining: 2m 51s
1474:	learn: 2.1435981	test: 2.2138837	best: 2.2133807 (1433)	total: 38.8s	remaining: 2m 51s
1476:	learn: 2.1422215	test: 2.2139173	best: 2.2133807 (1433)	total: 38.8s	remaining: 2m 51s
1478:	learn: 2.1419292	test: 2.2140003	best: 2.2133807 (1433)	total: 38.9s	remaining: 2m 51s
1480:	learn: 2.1418737	test: 2.2140257	best: 2.2133807 (1433)	total: 39s	remaining: 2m 51s
1482:	learn: 2.1413431	test: 2.2139101	best: 2.2133807 (1433)	total: 39s	remaining: 2m 51s
1484:	learn: 2.1411008	test: 2.2137565	best: 2.2133807 (1433)	total: 39.1s	remaining: 2m 51s
1486:	learn: 2.1406511	test: 2.2139086	best: 2.2133807 (1433)	total: 39.1s	remaining: 2m 51s
1488:	learn: 2.1404790	test: 2.2139465	best: 2.2133807 (1433)	total: 39.2s	remaining: 2m 51s
1490:	learn: 2.1404330	test: 2.2139153	best: 2.2133807 (1433)	total: 39.2s	remaining: 2m 51s
1492:	learn: 2.1402041	test: 2.2140301	best: 2.2133807 (1433)	total: 39.3s	remaining: 2m 51s
1494:	learn: 2.1400028	test: 2.2143094	best: 2.2133807 (1433)	total: 39.3s	remaining: 2m 51s
1496:	learn: 2.1396657	test: 2.2144496	best: 2.2133807 (1433)	total: 39.4s	remaining: 2m 51s
1498:	learn: 2.1391557	test: 2.2145510	best: 2.2133807 (1433)	total: 39.4s	remaining: 2m 51s
1500:	learn: 2.1386189	test: 2.2145807	best: 2.2133807 (1433)	total: 39.5s	remaining: 2m 51s
1502:	learn: 2.1385014	test: 2.2146673	best: 2.2133807 (1433)	total: 39.5s	remaining: 2m 50s
1504:	learn: 2.1383820	test: 2.2147072	best: 2.2133807 (1433)	total: 39.6s	remaining: 2m 50s
1506:	learn: 2.1381517	test: 2.2146811	best: 2.2133807 (1433)	total: 39.7s	remaining: 2m 50s
1508:	learn: 2.1376459	test: 2.2146610	best: 2.2133807 (1433)	total: 39.7s	remaining: 2m 50s
1510:	learn: 2.1375070	test: 2.2146671	best: 2.2133807 (1433)	total: 39.8s	remaining: 2m 50s
1512:	learn: 2.1370652	test: 2.2147590	best: 2.2133807 (1433)	total: 39.8s	remaining: 2m 50s
1514:	learn: 2.1369819	test: 2.2147413	best: 2.2133807 (1433)	total: 39.9s	remaining: 2m 50s
1516:	learn: 2.1368708	test: 2.2146385	best: 2.2133807 (1433)	total: 39.9s	remaining: 2m 50s
1518:	learn: 2.1366676	test: 2.2147752	best: 2.2133807 (1433)	total: 40s	remaining: 2m 50s
1520:	learn: 2.1364312	test: 2.2149780	best: 2.2133807 (1433)	total: 40s	remaining: 2m 50s
1522:	learn: 2.1363068	test: 2.2148531	best: 2.2133807 (1433)	total: 40.1s	remaining: 2m 50s
1524:	learn: 2.1360560	test: 2.2150098	best: 2.2133807 (1433)	total: 40.1s	remaining: 2m 50s
1526:	learn: 2.1359357	test: 2.2149829	best: 2.2133807 (1433)	total: 40.2s	remaining: 2m 50s
1528:	learn: 2.1357790	test: 2.2149922	best: 2.2133807 (1433)	total: 40.2s	remaining: 2m 50s
1530:	learn: 2.1351030	test: 2.2153256	best: 2.2133807 (1433)	total: 40.3s	remaining: 2m 50s
1532:	learn: 2.1341898	test: 2.2153842	best: 2.2133807 (1433)	total: 40.3s	remaining: 2m 50s
1534:	learn: 2.1338658	test: 2.2153885	best: 2.2133807 (1433)	total: 40.4s	remaining: 2m 50s
1536:	learn: 2.1335928	test: 2.2153266	best: 2.2133807 (1433)	total: 40.4s	remaining: 2m 50s
1538:	learn: 2.1335269	test: 2.2153052	best: 2.2133807 (1433)	total: 40.5s	remaining: 2m 50s
1540:	learn: 2.1332226	test: 2.2151257	best: 2.2133807 (1433)	total: 40.6s	remaining: 2m 50s
1542:	learn: 2.1330517	test: 2.2151550	best: 2.2133807 (1433)	total: 40.6s	remaining: 2m 49s
1544:	learn: 2.1325123	test: 2.2154313	best: 2.2133807 (1433)	total: 40.7s	remaining: 2m 49s
1546:	learn: 2.1316706	test: 2.2151844	best: 2.2133807 (1433)	total: 40.7s	remaining: 2m 49s
1548:	learn: 2.1315480	test: 2.2154220	best: 2.2133807 (1433)	total: 40.8s	remaining: 2m 49s
1550:	learn: 2.1301698	test: 2.2154483	best: 2.2133807 (1433)	total: 40.8s	remaining: 2m 49s
1552:	learn: 2.1297285	test: 2.2153912	best: 2.2133807 (1433)	total: 40.9s	remaining: 2m 49s
1554:	learn: 2.1294151	test: 2.2155018	best: 2.2133807 (1433)	total: 41s	remaining: 2m 49s
1556:	learn: 2.1292914	test: 2.2155076	best: 2.2133807 (1433)	total: 41s	remaining: 2m 49s
1558:	learn: 2.1291487	test: 2.2156053	best: 2.2133807 (1433)	total: 41.1s	remaining: 2m 49s
1560:	learn: 2.1289514	test: 2.2160052	best: 2.2133807 (1433)	total: 41.1s	remaining: 2m 49s
1562:	learn: 2.1273863	test: 2.2157028	best: 2.2133807 (1433)	total: 41.2s	remaining: 2m 49s
1564:	learn: 2.1270213	test: 2.2154958	best: 2.2133807 (1433)	total: 41.2s	remaining: 2m 49s
1566:	learn: 2.1268793	test: 2.2154605	best: 2.2133807 (1433)	total: 41.3s	remaining: 2m 49s
1568:	learn: 2.1267308	test: 2.2155230	best: 2.2133807 (1433)	total: 41.3s	remaining: 2m 49s
1570:	learn: 2.1262305	test: 2.2155083	best: 2.2133807 (1433)	total: 41.4s	remaining: 2m 49s
1572:	learn: 2.1260226	test: 2.2155367	best: 2.2133807 (1433)	total: 41.4s	remaining: 2m 49s
1574:	learn: 2.1258561	test: 2.2156490	best: 2.2133807 (1433)	total: 41.5s	remaining: 2m 49s
1576:	learn: 2.1254226	test: 2.2155991	best: 2.2133807 (1433)	total: 41.5s	remaining: 2m 49s
1578:	learn: 2.1252135	test: 2.2157630	best: 2.2133807 (1433)	total: 41.6s	remaining: 2m 49s
1580:	learn: 2.1251622	test: 2.2158615	best: 2.2133807 (1433)	total: 41.7s	remaining: 2m 49s
1582:	learn: 2.1249355	test: 2.2158799	best: 2.2133807 (1433)	total: 41.7s	remaining: 2m 49s
1584:	learn: 2.1248591	test: 2.2158882	best: 2.2133807 (1433)	total: 41.8s	remaining: 2m 49s
1586:	learn: 2.1241451	test: 2.2163806	best: 2.2133807 (1433)	total: 41.8s	remaining: 2m 48s
1588:	learn: 2.1239749	test: 2.2163102	best: 2.2133807 (1433)	total: 41.9s	remaining: 2m 48s
1590:	learn: 2.1233384	test: 2.2162697	best: 2.2133807 (1433)	total: 41.9s	remaining: 2m 48s
1592:	learn: 2.1231490	test: 2.2161979	best: 2.2133807 (1433)	total: 42s	remaining: 2m 48s
1594:	learn: 2.1224166	test: 2.2168284	best: 2.2133807 (1433)	total: 42s	remaining: 2m 48s
1596:	learn: 2.1218982	test: 2.2166448	best: 2.2133807 (1433)	total: 42.1s	remaining: 2m 48s
1598:	learn: 2.1213664	test: 2.2167041	best: 2.2133807 (1433)	total: 42.1s	remaining: 2m 48s
1600:	learn: 2.1207722	test: 2.2167001	best: 2.2133807 (1433)	total: 42.2s	remaining: 2m 48s
1602:	learn: 2.1205194	test: 2.2168191	best: 2.2133807 (1433)	total: 42.3s	remaining: 2m 48s
1604:	learn: 2.1200213	test: 2.2169096	best: 2.2133807 (1433)	total: 42.3s	remaining: 2m 48s
1606:	learn: 2.1196670	test: 2.2169259	best: 2.2133807 (1433)	total: 42.4s	remaining: 2m 48s
1608:	learn: 2.1192427	test: 2.2168746	best: 2.2133807 (1433)	total: 42.4s	remaining: 2m 48s
1610:	learn: 2.1191763	test: 2.2168927	best: 2.2133807 (1433)	total: 42.5s	remaining: 2m 48s
1612:	learn: 2.1188964	test: 2.2166567	best: 2.2133807 (1433)	total: 42.5s	remaining: 2m 48s
1614:	learn: 2.1186025	test: 2.2166524	best: 2.2133807 (1433)	total: 42.6s	remaining: 2m 48s
1616:	learn: 2.1185858	test: 2.2166578	best: 2.2133807 (1433)	total: 42.6s	remaining: 2m 48s
1618:	learn: 2.1176642	test: 2.2165314	best: 2.2133807 (1433)	total: 42.7s	remaining: 2m 48s
1620:	learn: 2.1171013	test: 2.2163507	best: 2.2133807 (1433)	total: 42.8s	remaining: 2m 48s
1622:	learn: 2.1169249	test: 2.2164884	best: 2.2133807 (1433)	total: 42.8s	remaining: 2m 48s
1624:	learn: 2.1167070	test: 2.2164722	best: 2.2133807 (1433)	total: 42.9s	remaining: 2m 48s
1626:	learn: 2.1163664	test: 2.2165483	best: 2.2133807 (1433)	total: 42.9s	remaining: 2m 48s
1628:	learn: 2.1161851	test: 2.2167402	best: 2.2133807 (1433)	total: 43s	remaining: 2m 48s
1630:	learn: 2.1159175	test: 2.2166951	best: 2.2133807 (1433)	total: 43s	remaining: 2m 47s
1632:	learn: 2.1154491	test: 2.2167102	best: 2.2133807 (1433)	total: 43.1s	remaining: 2m 47s
1634:	learn: 2.1152643	test: 2.2166657	best: 2.2133807 (1433)	total: 43.1s	remaining: 2m 47s
1636:	learn: 2.1150777	test: 2.2167100	best: 2.2133807 (1433)	total: 43.2s	remaining: 2m 47s
1638:	learn: 2.1149859	test: 2.2168145	best: 2.2133807 (1433)	total: 43.2s	remaining: 2m 47s
1640:	learn: 2.1147497	test: 2.2169758	best: 2.2133807 (1433)	total: 43.3s	remaining: 2m 47s
1642:	learn: 2.1143066	test: 2.2167931	best: 2.2133807 (1433)	total: 43.3s	remaining: 2m 47s
1644:	learn: 2.1139391	test: 2.2168331	best: 2.2133807 (1433)	total: 43.4s	remaining: 2m 47s
1646:	learn: 2.1138049	test: 2.2168283	best: 2.2133807 (1433)	total: 43.4s	remaining: 2m 47s
1648:	learn: 2.1134589	test: 2.2166591	best: 2.2133807 (1433)	total: 43.5s	remaining: 2m 47s
1650:	learn: 2.1131977	test: 2.2167813	best: 2.2133807 (1433)	total: 43.5s	remaining: 2m 47s
1652:	learn: 2.1128272	test: 2.2167216	best: 2.2133807 (1433)	total: 43.6s	remaining: 2m 47s
1654:	learn: 2.1127059	test: 2.2167172	best: 2.2133807 (1433)	total: 43.7s	remaining: 2m 47s
1656:	learn: 2.1125227	test: 2.2168117	best: 2.2133807 (1433)	total: 43.7s	remaining: 2m 47s
1658:	learn: 2.1124348	test: 2.2167637	best: 2.2133807 (1433)	total: 43.8s	remaining: 2m 47s
1660:	learn: 2.1120518	test: 2.2167041	best: 2.2133807 (1433)	total: 43.8s	remaining: 2m 47s
1662:	learn: 2.1118641	test: 2.2166072	best: 2.2133807 (1433)	total: 43.9s	remaining: 2m 47s
1664:	learn: 2.1112053	test: 2.2166701	best: 2.2133807 (1433)	total: 43.9s	remaining: 2m 47s
1666:	learn: 2.1110279	test: 2.2166214	best: 2.2133807 (1433)	total: 44s	remaining: 2m 47s
1668:	learn: 2.1109956	test: 2.2166148	best: 2.2133807 (1433)	total: 44s	remaining: 2m 47s
1670:	learn: 2.1106505	test: 2.2167319	best: 2.2133807 (1433)	total: 44.1s	remaining: 2m 47s
1672:	learn: 2.1102161	test: 2.2167706	best: 2.2133807 (1433)	total: 44.2s	remaining: 2m 46s
1674:	learn: 2.1100418	test: 2.2168141	best: 2.2133807 (1433)	total: 44.2s	remaining: 2m 46s
1676:	learn: 2.1099993	test: 2.2167669	best: 2.2133807 (1433)	total: 44.3s	remaining: 2m 46s
1678:	learn: 2.1097879	test: 2.2167990	best: 2.2133807 (1433)	total: 44.3s	remaining: 2m 46s
1680:	learn: 2.1095550	test: 2.2168662	best: 2.2133807 (1433)	total: 44.4s	remaining: 2m 46s
1682:	learn: 2.1094071	test: 2.2169120	best: 2.2133807 (1433)	total: 44.4s	remaining: 2m 46s
1684:	learn: 2.1092353	test: 2.2168429	best: 2.2133807 (1433)	total: 44.5s	remaining: 2m 46s
1686:	learn: 2.1091700	test: 2.2168785	best: 2.2133807 (1433)	total: 44.6s	remaining: 2m 46s
1688:	learn: 2.1088413	test: 2.2167786	best: 2.2133807 (1433)	total: 44.6s	remaining: 2m 46s
1690:	learn: 2.1084285	test: 2.2170379	best: 2.2133807 (1433)	total: 44.7s	remaining: 2m 46s
1692:	learn: 2.1082539	test: 2.2172313	best: 2.2133807 (1433)	total: 44.7s	remaining: 2m 46s
1694:	learn: 2.1079285	test: 2.2171574	best: 2.2133807 (1433)	total: 44.8s	remaining: 2m 46s
1696:	learn: 2.1076022	test: 2.2176235	best: 2.2133807 (1433)	total: 44.8s	remaining: 2m 46s
1698:	learn: 2.1074210	test: 2.2178718	best: 2.2133807 (1433)	total: 44.9s	remaining: 2m 46s
1700:	learn: 2.1072077	test: 2.2178526	best: 2.2133807 (1433)	total: 44.9s	remaining: 2m 46s
1702:	learn: 2.1070756	test: 2.2177726	best: 2.2133807 (1433)	total: 45s	remaining: 2m 46s
1704:	learn: 2.1068673	test: 2.2177309	best: 2.2133807 (1433)	total: 45s	remaining: 2m 46s
1706:	learn: 2.1065192	test: 2.2176168	best: 2.2133807 (1433)	total: 45.1s	remaining: 2m 46s
1708:	learn: 2.1063914	test: 2.2174584	best: 2.2133807 (1433)	total: 45.1s	remaining: 2m 46s
1710:	learn: 2.1060010	test: 2.2175298	best: 2.2133807 (1433)	total: 45.2s	remaining: 2m 46s
1712:	learn: 2.1058080	test: 2.2175425	best: 2.2133807 (1433)	total: 45.2s	remaining: 2m 45s
1714:	learn: 2.1055128	test: 2.2173537	best: 2.2133807 (1433)	total: 45.3s	remaining: 2m 45s
1716:	learn: 2.1051679	test: 2.2175149	best: 2.2133807 (1433)	total: 45.3s	remaining: 2m 45s
1718:	learn: 2.1048660	test: 2.2175229	best: 2.2133807 (1433)	total: 45.4s	remaining: 2m 45s
1720:	learn: 2.1047099	test: 2.2178264	best: 2.2133807 (1433)	total: 45.4s	remaining: 2m 45s
1722:	learn: 2.1045924	test: 2.2179405	best: 2.2133807 (1433)	total: 45.5s	remaining: 2m 45s
1724:	learn: 2.1043486	test: 2.2179963	best: 2.2133807 (1433)	total: 45.5s	remaining: 2m 45s
1726:	learn: 2.1041868	test: 2.2179530	best: 2.2133807 (1433)	total: 45.6s	remaining: 2m 45s
1728:	learn: 2.1037333	test: 2.2182736	best: 2.2133807 (1433)	total: 45.6s	remaining: 2m 45s
1730:	learn: 2.1032962	test: 2.2184658	best: 2.2133807 (1433)	total: 45.7s	remaining: 2m 45s
1732:	learn: 2.1031704	test: 2.2186196	best: 2.2133807 (1433)	total: 45.7s	remaining: 2m 45s
1734:	learn: 2.1030589	test: 2.2185699	best: 2.2133807 (1433)	total: 45.8s	remaining: 2m 45s
1736:	learn: 2.1029396	test: 2.2185258	best: 2.2133807 (1433)	total: 45.8s	remaining: 2m 45s
1738:	learn: 2.1026188	test: 2.2187772	best: 2.2133807 (1433)	total: 45.9s	remaining: 2m 45s
1740:	learn: 2.1021376	test: 2.2187388	best: 2.2133807 (1433)	total: 45.9s	remaining: 2m 45s
1742:	learn: 2.1016550	test: 2.2188459	best: 2.2133807 (1433)	total: 46s	remaining: 2m 45s
1744:	learn: 2.1000430	test: 2.2188917	best: 2.2133807 (1433)	total: 46s	remaining: 2m 44s
1746:	learn: 2.0998245	test: 2.2190074	best: 2.2133807 (1433)	total: 46.1s	remaining: 2m 44s
1748:	learn: 2.0984635	test: 2.2188044	best: 2.2133807 (1433)	total: 46.1s	remaining: 2m 44s
1750:	learn: 2.0981770	test: 2.2189612	best: 2.2133807 (1433)	total: 46.2s	remaining: 2m 44s
1752:	learn: 2.0979454	test: 2.2189281	best: 2.2133807 (1433)	total: 46.2s	remaining: 2m 44s
1754:	learn: 2.0975278	test: 2.2189757	best: 2.2133807 (1433)	total: 46.3s	remaining: 2m 44s
1756:	learn: 2.0974667	test: 2.2189899	best: 2.2133807 (1433)	total: 46.3s	remaining: 2m 44s
1758:	learn: 2.0971750	test: 2.2187568	best: 2.2133807 (1433)	total: 46.4s	remaining: 2m 44s
1760:	learn: 2.0970061	test: 2.2188531	best: 2.2133807 (1433)	total: 46.5s	remaining: 2m 44s
1762:	learn: 2.0963113	test: 2.2188273	best: 2.2133807 (1433)	total: 46.5s	remaining: 2m 44s
1764:	learn: 2.0960983	test: 2.2188298	best: 2.2133807 (1433)	total: 46.6s	remaining: 2m 44s
1766:	learn: 2.0959246	test: 2.2188553	best: 2.2133807 (1433)	total: 46.6s	remaining: 2m 44s
1768:	learn: 2.0958570	test: 2.2189443	best: 2.2133807 (1433)	total: 46.7s	remaining: 2m 44s
1770:	learn: 2.0957937	test: 2.2189675	best: 2.2133807 (1433)	total: 46.7s	remaining: 2m 44s
1772:	learn: 2.0950692	test: 2.2186340	best: 2.2133807 (1433)	total: 46.8s	remaining: 2m 44s
1774:	learn: 2.0949028	test: 2.2186312	best: 2.2133807 (1433)	total: 46.8s	remaining: 2m 44s
1776:	learn: 2.0943527	test: 2.2186558	best: 2.2133807 (1433)	total: 46.9s	remaining: 2m 44s
1778:	learn: 2.0934362	test: 2.2189315	best: 2.2133807 (1433)	total: 46.9s	remaining: 2m 44s
1780:	learn: 2.0928994	test: 2.2188489	best: 2.2133807 (1433)	total: 47s	remaining: 2m 44s
1782:	learn: 2.0922152	test: 2.2188101	best: 2.2133807 (1433)	total: 47s	remaining: 2m 44s
1784:	learn: 2.0916621	test: 2.2188406	best: 2.2133807 (1433)	total: 47.1s	remaining: 2m 44s
1786:	learn: 2.0914689	test: 2.2186830	best: 2.2133807 (1433)	total: 47.2s	remaining: 2m 43s
1788:	learn: 2.0911672	test: 2.2186190	best: 2.2133807 (1433)	total: 47.2s	remaining: 2m 43s
1790:	learn: 2.0908374	test: 2.2186478	best: 2.2133807 (1433)	total: 47.3s	remaining: 2m 43s
1792:	learn: 2.0905446	test: 2.2187249	best: 2.2133807 (1433)	total: 47.3s	remaining: 2m 43s
1794:	learn: 2.0904438	test: 2.2183732	best: 2.2133807 (1433)	total: 47.4s	remaining: 2m 43s
1796:	learn: 2.0899459	test: 2.2182114	best: 2.2133807 (1433)	total: 47.4s	remaining: 2m 43s
1798:	learn: 2.0897115	test: 2.2182591	best: 2.2133807 (1433)	total: 47.5s	remaining: 2m 43s
1800:	learn: 2.0894917	test: 2.2180988	best: 2.2133807 (1433)	total: 47.5s	remaining: 2m 43s
1802:	learn: 2.0892646	test: 2.2182871	best: 2.2133807 (1433)	total: 47.6s	remaining: 2m 43s
1804:	learn: 2.0890696	test: 2.2182821	best: 2.2133807 (1433)	total: 47.6s	remaining: 2m 43s
1806:	learn: 2.0887946	test: 2.2181681	best: 2.2133807 (1433)	total: 47.7s	remaining: 2m 43s
1808:	learn: 2.0881146	test: 2.2185233	best: 2.2133807 (1433)	total: 47.7s	remaining: 2m 43s
1810:	learn: 2.0875686	test: 2.2180469	best: 2.2133807 (1433)	total: 47.8s	remaining: 2m 43s
1812:	learn: 2.0874828	test: 2.2180356	best: 2.2133807 (1433)	total: 47.9s	remaining: 2m 43s
1814:	learn: 2.0871639	test: 2.2180150	best: 2.2133807 (1433)	total: 47.9s	remaining: 2m 43s
1816:	learn: 2.0870024	test: 2.2178952	best: 2.2133807 (1433)	total: 48s	remaining: 2m 43s
1818:	learn: 2.0866337	test: 2.2180757	best: 2.2133807 (1433)	total: 48s	remaining: 2m 43s
1820:	learn: 2.0862102	test: 2.2178992	best: 2.2133807 (1433)	total: 48.1s	remaining: 2m 43s
1822:	learn: 2.0860513	test: 2.2177321	best: 2.2133807 (1433)	total: 48.1s	remaining: 2m 43s
1824:	learn: 2.0859814	test: 2.2177095	best: 2.2133807 (1433)	total: 48.2s	remaining: 2m 43s
1826:	learn: 2.0857023	test: 2.2176400	best: 2.2133807 (1433)	total: 48.3s	remaining: 2m 43s
1828:	learn: 2.0855946	test: 2.2177762	best: 2.2133807 (1433)	total: 48.3s	remaining: 2m 42s
1830:	learn: 2.0853615	test: 2.2176758	best: 2.2133807 (1433)	total: 48.3s	remaining: 2m 42s
1832:	learn: 2.0852357	test: 2.2176744	best: 2.2133807 (1433)	total: 48.4s	remaining: 2m 42s
1834:	learn: 2.0850998	test: 2.2175935	best: 2.2133807 (1433)	total: 48.5s	remaining: 2m 42s
1836:	learn: 2.0846760	test: 2.2174814	best: 2.2133807 (1433)	total: 48.5s	remaining: 2m 42s
1838:	learn: 2.0839724	test: 2.2176719	best: 2.2133807 (1433)	total: 48.6s	remaining: 2m 42s
1840:	learn: 2.0832457	test: 2.2178385	best: 2.2133807 (1433)	total: 48.6s	remaining: 2m 42s
1842:	learn: 2.0830096	test: 2.2176367	best: 2.2133807 (1433)	total: 48.7s	remaining: 2m 42s
1844:	learn: 2.0827872	test: 2.2175105	best: 2.2133807 (1433)	total: 48.7s	remaining: 2m 42s
1846:	learn: 2.0826329	test: 2.2175579	best: 2.2133807 (1433)	total: 48.8s	remaining: 2m 42s
1848:	learn: 2.0821098	test: 2.2167353	best: 2.2133807 (1433)	total: 48.8s	remaining: 2m 42s
1850:	learn: 2.0816314	test: 2.2162743	best: 2.2133807 (1433)	total: 48.9s	remaining: 2m 42s
1852:	learn: 2.0814814	test: 2.2163497	best: 2.2133807 (1433)	total: 48.9s	remaining: 2m 42s
1854:	learn: 2.0812301	test: 2.2161839	best: 2.2133807 (1433)	total: 49s	remaining: 2m 42s
1856:	learn: 2.0810467	test: 2.2161243	best: 2.2133807 (1433)	total: 49.1s	remaining: 2m 42s
1858:	learn: 2.0807751	test: 2.2163141	best: 2.2133807 (1433)	total: 49.1s	remaining: 2m 42s
1860:	learn: 2.0802613	test: 2.2164809	best: 2.2133807 (1433)	total: 49.2s	remaining: 2m 42s
1862:	learn: 2.0799178	test: 2.2165635	best: 2.2133807 (1433)	total: 49.2s	remaining: 2m 42s
1864:	learn: 2.0796691	test: 2.2164092	best: 2.2133807 (1433)	total: 49.3s	remaining: 2m 42s
1866:	learn: 2.0794782	test: 2.2164164	best: 2.2133807 (1433)	total: 49.3s	remaining: 2m 42s
1868:	learn: 2.0791274	test: 2.2164162	best: 2.2133807 (1433)	total: 49.4s	remaining: 2m 42s
1870:	learn: 2.0785926	test: 2.2164291	best: 2.2133807 (1433)	total: 49.5s	remaining: 2m 41s
1872:	learn: 2.0785356	test: 2.2164454	best: 2.2133807 (1433)	total: 49.5s	remaining: 2m 41s
1874:	learn: 2.0781246	test: 2.2163962	best: 2.2133807 (1433)	total: 49.6s	remaining: 2m 41s
1876:	learn: 2.0779868	test: 2.2163734	best: 2.2133807 (1433)	total: 49.6s	remaining: 2m 41s
1878:	learn: 2.0778354	test: 2.2163548	best: 2.2133807 (1433)	total: 49.7s	remaining: 2m 41s
1880:	learn: 2.0774647	test: 2.2161153	best: 2.2133807 (1433)	total: 49.7s	remaining: 2m 41s
1882:	learn: 2.0772928	test: 2.2161577	best: 2.2133807 (1433)	total: 49.8s	remaining: 2m 41s
1884:	learn: 2.0771379	test: 2.2159752	best: 2.2133807 (1433)	total: 49.8s	remaining: 2m 41s
1886:	learn: 2.0770816	test: 2.2159685	best: 2.2133807 (1433)	total: 49.9s	remaining: 2m 41s
1888:	learn: 2.0769629	test: 2.2158854	best: 2.2133807 (1433)	total: 50s	remaining: 2m 41s
1890:	learn: 2.0767333	test: 2.2157726	best: 2.2133807 (1433)	total: 50s	remaining: 2m 41s
1892:	learn: 2.0765138	test: 2.2156827	best: 2.2133807 (1433)	total: 50.1s	remaining: 2m 41s
1894:	learn: 2.0761470	test: 2.2158858	best: 2.2133807 (1433)	total: 50.1s	remaining: 2m 41s
1896:	learn: 2.0755202	test: 2.2148326	best: 2.2133807 (1433)	total: 50.2s	remaining: 2m 41s
1898:	learn: 2.0753735	test: 2.2149049	best: 2.2133807 (1433)	total: 50.2s	remaining: 2m 41s
1900:	learn: 2.0750506	test: 2.2150271	best: 2.2133807 (1433)	total: 50.3s	remaining: 2m 41s
1902:	learn: 2.0749413	test: 2.2150955	best: 2.2133807 (1433)	total: 50.3s	remaining: 2m 41s
1904:	learn: 2.0748590	test: 2.2149790	best: 2.2133807 (1433)	total: 50.4s	remaining: 2m 41s
1906:	learn: 2.0747494	test: 2.2148978	best: 2.2133807 (1433)	total: 50.4s	remaining: 2m 41s
1908:	learn: 2.0743479	test: 2.2153460	best: 2.2133807 (1433)	total: 50.5s	remaining: 2m 41s
1910:	learn: 2.0739797	test: 2.2152825	best: 2.2133807 (1433)	total: 50.6s	remaining: 2m 41s
1912:	learn: 2.0738103	test: 2.2153011	best: 2.2133807 (1433)	total: 50.6s	remaining: 2m 41s
1914:	learn: 2.0735026	test: 2.2150761	best: 2.2133807 (1433)	total: 50.7s	remaining: 2m 40s
1916:	learn: 2.0730609	test: 2.2145698	best: 2.2133807 (1433)	total: 50.7s	remaining: 2m 40s
1918:	learn: 2.0726296	test: 2.2144750	best: 2.2133807 (1433)	total: 50.8s	remaining: 2m 40s
1920:	learn: 2.0722862	test: 2.2147829	best: 2.2133807 (1433)	total: 50.8s	remaining: 2m 40s
1922:	learn: 2.0717825	test: 2.2145992	best: 2.2133807 (1433)	total: 50.9s	remaining: 2m 40s
1924:	learn: 2.0712879	test: 2.2146716	best: 2.2133807 (1433)	total: 50.9s	remaining: 2m 40s
1926:	learn: 2.0710907	test: 2.2147340	best: 2.2133807 (1433)	total: 51s	remaining: 2m 40s
1928:	learn: 2.0707983	test: 2.2147183	best: 2.2133807 (1433)	total: 51s	remaining: 2m 40s
1930:	learn: 2.0704413	test: 2.2153718	best: 2.2133807 (1433)	total: 51.1s	remaining: 2m 40s
1932:	learn: 2.0699433	test: 2.2155413	best: 2.2133807 (1433)	total: 51.1s	remaining: 2m 40s
Stopped by overfitting detector  (500 iterations wait)

bestTest = 2.213380684
bestIteration = 1433

Shrink model to first 1434 iterations.
</pre>
<pre>
<catboost.core.CatBoostRegressor at 0x7f9f00900d90>
</pre>

```python
y_pred = model.predict(valid_pool)
```


```python
y_pred_rounded = np.round(y_pred).astype(int)
```


```python
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
rmse
```

<pre>
2.213380683573479
</pre>
## 3.1 특성 중요도



```python
feature_importances = model.get_feature_importance()

sorted_indices = feature_importances.argsort()[::-1]
sorted_features = X_train.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

print("특성 중요도 (높은 순서대로):")
for feature_name, importance in zip(sorted_features, sorted_importances):
    print(f"{feature_name}: {importance}")
```

<pre>
특성 중요도 (높은 순서대로):
quality: 41.13160425519476
duration: 16.27470560409945
duration_quality_ratio: 10.28232648713925
subcontinent: 9.15194908014832
new: 4.103057376135241
OS: 3.04994478859505
traffic_medium: 2.727123465142591
traffic_source: 2.256529514515024
device: 2.004816411823524
browser: 1.6585721948997443
referral_path: 1.6214543670562813
transaction_revenue: 1.6177482405391284
transaction: 1.23824997940051
country: 1.0482169085561337
bounced: 0.9839548440413372
keyword: 0.4681961150808162
continent: 0.3815503676326892
</pre>
# 4. 테스트 셋



```python
test_pool = Pool(data=df_test, cat_features=categorical_features)
test_pred = model.predict(test_pool)
```


```python
test_pred = [0 if i < 0 else i for i in test_pred]
```


```python
submit = pd.read_csv('sample_submission.csv')
submit["TARGET"] = test_pred
```


```python
submit.to_csv("sub_2_21.csv", index=False)
```
