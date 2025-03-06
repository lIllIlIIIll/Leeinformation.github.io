---
layout: single
title:  "시황 분석 LSTM 모델 훈련"
categories: AI
tag: [python, coding, Machine Running]
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


이전 Kiwoom 자동 매매 프로그램 구현 포스팅들 중 시황분석 페이지에서 투자 점수 부분을 구현하지 않고 남겨두었다.



해당 부분을 LSTM 모델로 간단하게 오늘 투자 위험 여부를 판단하는 모델을 훈련시켜 적용하도록 구현한다.


# 데이터 전처리


데이터셋은 **Investing.com** 사이트에서 과거 5년간의 시황 데이터(환율, 미국 국채 10년, 서부 텍사스유 등등)를 이용하였다.



```python
import os
import pandas as pd
import glob

def preprocessing(dataset, filename) :
    # 날짜 기준 오름차순 정렬
    dataset["날짜"] = pd.to_datetime(dataset["날짜"])
    dataset = dataset.sort_values(by="날짜", ascending=True).reset_index(drop=True)
    
    if "거래량" in dataset.columns :
        dataset = dataset.drop(["거래량"], axis=1)
    
    # 각 지표 이름 붙이기
    prefix = os.path.splitext(os.path.basename(filename))[0]
    dataset.columns = [f"{prefix}_{col}" if col != "날짜" else col for col in dataset.columns]
    
    return dataset
```


```python
import numpy as np

# % 변환 함수
def convert_percent(x) :
        if isinstance(x, str) and "%" in x :
            return float(x.replace("%", "")) / 100.0
        return x

def merge_preprocessing(dataset_list) :
    if len(dataset_list) >= 2 :
        standard_data = dataset_list[0]
    else :
        standard_data = dataset_list
    
    # 같은 날짜 데이터 병합
    if len(dataset_list) >= 2 :
        for df in dataset_list[1:] :
            date_column = [col for col in df.columns if '날짜' in col]
            standard_data = pd.merge(standard_data, df, on=date_column, how="outer")
    
    # 주식장이 쉬는 날은 변동을 NaN 에서 0.00%로 변경    
    mask = standard_data.columns.str.endswith("_변동 %")
    standard_data.loc[:, mask] = standard_data.loc[:, mask].apply(lambda col: col.fillna("0.00%"))
    
    # 나머지 NaN값들은 이전 날의 데이터 쓰기(주말 or 공휴일일 가능성이 높으므로)
    standard_data.fillna(method="ffill", inplace=True)
    
    # 데이터 날짜 기준 정렬
    standard_data = standard_data.sort_values(by="날짜", ascending=True).reset_index(drop=True)
    
    # %로 표현된 값들 수치형 데이터로 표현
    percent_columns = [col for col in standard_data.columns if col.endswith("_변동 %")]
    standard_data[percent_columns] = standard_data[percent_columns].applymap(convert_percent)
    
    # ","가 포함된 object 타입 수치형 데이터로 변환
    standard_data = standard_data.applymap(lambda x: float(x.replace(",", "")) if isinstance(x, str) else x)
    
    # 투자 여부 레이블 추가(국장이므로 코스닥과 코스피 변동률을 기준으로 4분할하여 지정)
    if len(standard_data) >= 2 :
        standard_point = standard_data["코스닥_변동 %"] + standard_data["코스피_변동 %"]
        quartiles = pd.qcut(standard_point, 4, labels=False)
        standard_data["투자 여부"] = quartiles
    
    return standard_data
```


```python
import re

csv_files = glob.glob(os.path.join("*.csv"))
tr_csv_files = [file for file in csv_files if re.search("[가-힣]", file)]

df_list = []

for file in tr_csv_files :
    df = pd.read_csv(file)
    pro_df = preprocessing(df, file)
    df_list.append(pro_df)
```


```python
data = merge_preprocessing(df_list)
```

<pre>
/tmp/ipykernel_61336/2289762405.py:26: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  standard_data.fillna(method="ffill", inplace=True)
/tmp/ipykernel_61336/2289762405.py:33: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
  standard_data[percent_columns] = standard_data[percent_columns].applymap(convert_percent)
/tmp/ipykernel_61336/2289762405.py:36: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
  standard_data = standard_data.applymap(lambda x: float(x.replace(",", "")) if isinstance(x, str) else x)
</pre>
# 모델 훈련 및 저장



```python
from sklearn.preprocessing import StandardScaler
import joblib

def train_scaler(dataset, save_path="scaler.pkl") :
    if "투자 여부" in dataset.columns:
        numeric_data = dataset.drop(columns=["투자 여부"]).select_dtypes(include=[np.number])
    else:
        numeric_data = dataset.select_dtypes(include=[np.number])
    
    scaler = StandardScaler()
    scaler.fit(numeric_data)
    
    joblib.dump(scaler, save_path)
    
    return dataset
```


```python
def scale_data(dataset, scaler_path="scaler.pkl"):
    scaler = joblib.load(scaler_path)
    numeric_data = dataset.select_dtypes(include=[np.number])
    
    scaled_data = scaler.transform(numeric_data)
    
    dataset[numeric_data.columns] = scaled_data
    
    return dataset
```


```python
data = train_scaler(data)
```


```python
import numpy as np

data.set_index("날짜", inplace=True)

def create_sequence(df, seq_length, target_column="투자 여부") :
    sequence = []
    labels = []
    
    for i in range(len(df) - seq_length) :
        seq = df.iloc[i:i+seq_length].drop(columns=[target_column]).values
        label = df.iloc[i+seq_length][target_column]
        sequence.append(seq)
        labels.append(label)
    return np.array(sequence), np.array(labels)
```


```python
seq_length = 14

X, y = create_sequence(data, seq_length)
```


```python
import torch

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```


```python
import torch.nn as nn
import torch.optim as optim

class LSTMAutoencoder(nn.Module) :
    def __init__(self, input_dim, hidden_dim, latent_dim) :
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x) :
        _, (h_n, _) = self.encoder(x)
        z = self.latent(h_n[-1])
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(x)
        reconstructed = self.output_layer(decoded)
        
        return reconstructed, z
```


```python
# 미래 값 예측 모델

class FuturePredictor(nn.Module) :
    def __init__(self, latent_dim, prediction_dim) :
        super(FuturePredictor, self).__init__()
        self.lstm = nn.LSTM(latent_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, prediction_dim)
        
    def forward(self, z) :
        lstm_out, _ = self.lstm(z)
        future_values = self.fc(lstm_out[:, -1, :])
        return future_values
```


```python
input_dim = X.shape[-1]
hidden_dim = 64
latent_dim = 35
num_classes = 4
learning_rate = 1e-3

autoencoder = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)
predictor = FuturePredictor(latent_dim, num_classes)
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(list(autoencoder.parameters()) + list(predictor.parameters()), lr=learning_rate)
```


```python
num_epochs = 1000

for epoch in range(num_epochs) :
    optimizer.zero_grad()
    
    reconstructed, latent = autoencoder(X)
    future_values = predictor(latent)
    
    reconstruction_loss = reconstruction_criterion(reconstructed, X)
    prediction_loss = classification_criterion(future_values, y)
    
    loss = reconstruction_loss + prediction_loss
    
    loss = reconstruction_loss + prediction_loss
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0 :
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
with torch.no_grad() :
    _, latent = autoencoder(X)
    predicted_future_values = predictor(latent)
    predicted_classes = torch.argmax(predicted_future_values, dim=1)
```

<pre>
Epoch [10/1000], Loss: 1.2755
Epoch [20/1000], Loss: 1.1931
Epoch [30/1000], Loss: 1.1400
Epoch [40/1000], Loss: 1.0765
Epoch [50/1000], Loss: 1.0346
Epoch [60/1000], Loss: 0.9718
Epoch [70/1000], Loss: 0.9040
Epoch [80/1000], Loss: 0.8933
Epoch [90/1000], Loss: 0.8503
Epoch [100/1000], Loss: 0.7929
Epoch [110/1000], Loss: 0.7393
Epoch [120/1000], Loss: 0.6981
Epoch [130/1000], Loss: 0.6678
Epoch [140/1000], Loss: 0.6317
Epoch [150/1000], Loss: 0.6602
Epoch [160/1000], Loss: 0.6030
Epoch [170/1000], Loss: 0.5782
Epoch [180/1000], Loss: 0.5650
Epoch [190/1000], Loss: 0.5420
Epoch [200/1000], Loss: 0.4929
Epoch [210/1000], Loss: 0.5673
Epoch [220/1000], Loss: 0.5614
Epoch [230/1000], Loss: 0.5323
Epoch [240/1000], Loss: 0.4475
Epoch [250/1000], Loss: 0.4171
Epoch [260/1000], Loss: 0.3954
Epoch [270/1000], Loss: 0.3862
Epoch [280/1000], Loss: 0.3540
Epoch [290/1000], Loss: 0.3371
Epoch [300/1000], Loss: 0.7158
Epoch [310/1000], Loss: 0.4441
Epoch [320/1000], Loss: 0.3782
Epoch [330/1000], Loss: 0.3601
Epoch [340/1000], Loss: 0.3244
Epoch [350/1000], Loss: 0.2993
Epoch [360/1000], Loss: 0.2804
Epoch [370/1000], Loss: 0.2644
Epoch [380/1000], Loss: 0.2520
Epoch [390/1000], Loss: 0.3947
Epoch [400/1000], Loss: 0.4152
Epoch [410/1000], Loss: 0.3631
Epoch [420/1000], Loss: 0.3088
Epoch [430/1000], Loss: 0.2575
Epoch [440/1000], Loss: 0.2317
Epoch [450/1000], Loss: 0.2161
Epoch [460/1000], Loss: 0.2012
Epoch [470/1000], Loss: 0.1907
Epoch [480/1000], Loss: 0.1820
Epoch [490/1000], Loss: 0.1746
Epoch [500/1000], Loss: 0.1682
Epoch [510/1000], Loss: 0.1624
Epoch [520/1000], Loss: 0.1571
Epoch [530/1000], Loss: 0.1521
Epoch [540/1000], Loss: 0.1475
Epoch [550/1000], Loss: 0.1432
Epoch [560/1000], Loss: 0.1391
Epoch [570/1000], Loss: 0.1352
Epoch [580/1000], Loss: 0.1318
Epoch [590/1000], Loss: 0.2836
Epoch [600/1000], Loss: 0.7352
Epoch [610/1000], Loss: 0.4651
Epoch [620/1000], Loss: 0.3171
Epoch [630/1000], Loss: 0.2677
Epoch [640/1000], Loss: 0.3188
Epoch [650/1000], Loss: 0.2114
Epoch [660/1000], Loss: 0.1836
Epoch [670/1000], Loss: 0.1695
Epoch [680/1000], Loss: 0.1591
Epoch [690/1000], Loss: 0.1516
Epoch [700/1000], Loss: 0.1453
Epoch [710/1000], Loss: 0.1394
Epoch [720/1000], Loss: 0.1344
Epoch [730/1000], Loss: 0.1294
Epoch [740/1000], Loss: 0.1246
Epoch [750/1000], Loss: 0.1200
Epoch [760/1000], Loss: 0.1155
Epoch [770/1000], Loss: 0.1115
Epoch [780/1000], Loss: 0.1078
Epoch [790/1000], Loss: 0.1041
Epoch [800/1000], Loss: 0.1009
Epoch [810/1000], Loss: 0.0980
Epoch [820/1000], Loss: 0.0953
Epoch [830/1000], Loss: 0.0927
Epoch [840/1000], Loss: 0.0901
Epoch [850/1000], Loss: 0.0876
Epoch [860/1000], Loss: 0.0851
Epoch [870/1000], Loss: 0.0820
Epoch [880/1000], Loss: 0.0793
Epoch [890/1000], Loss: 0.0766
Epoch [900/1000], Loss: 0.0739
Epoch [910/1000], Loss: 0.0712
Epoch [920/1000], Loss: 0.0688
Epoch [930/1000], Loss: 0.0664
Epoch [940/1000], Loss: 0.0641
Epoch [950/1000], Loss: 0.0619
Epoch [960/1000], Loss: 0.0597
Epoch [970/1000], Loss: 0.0575
Epoch [980/1000], Loss: 0.0554
Epoch [990/1000], Loss: 0.0533
Epoch [1000/1000], Loss: 0.0513
</pre>

```python
with torch.no_grad():
    pre_data = data.drop("투자 여부", axis=1)
    last_14_days = pre_data.iloc[0:5].values

    last_14_days_tensor = torch.tensor(last_14_days, dtype=torch.float32).unsqueeze(0)
    
    reconstructed, latent = autoencoder(last_14_days_tensor)
    predicted_future_values = predictor(latent)
    
    predicted_class = torch.argmax(predicted_future_values, dim=1).item()
    
    print(predicted_class)
```

<pre>
1
</pre>

```python
torch.save(autoencoder.state_dict(), "autoencoder.pt")
torch.save(predictor.state_dict(), "predictor.pt")
```

# 실시간 당일 주가 정보들을 바탕으로 투자 여부 예측 (테스트)



```python
autoencoder.load_state_dict(torch.load("autoencoder.pt"))
predictor.load_state_dict(torch.load("predictor.pt"))

autoencoder.eval()
predictor.eval()
```

<pre>
/tmp/ipykernel_1426/2871261793.py:1: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  autoencoder.load_state_dict(torch.load("autoencoder.pt"))
/tmp/ipykernel_1426/2871261793.py:2: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  predictor.load_state_dict(torch.load("predictor.pt"))
</pre>
<pre>
FuturePredictor(
  (lstm): LSTM(35, 64, batch_first=True)
  (fc): Linear(in_features=64, out_features=4, bias=True)
)
</pre>

```python
from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')

today_data = pd.read_csv(f"{today}_trdata.csv")

today_data["날짜"] = pd.to_datetime(today_data["날짜"])
```


```python
processing_today_data = merge_preprocessing(today_data)
```

<pre>
/tmp/ipykernel_1426/2289762405.py:26: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  standard_data.fillna(method="ffill", inplace=True)
/tmp/ipykernel_1426/2289762405.py:33: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
  standard_data[percent_columns] = standard_data[percent_columns].applymap(convert_percent)
/tmp/ipykernel_1426/2289762405.py:36: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
  standard_data = standard_data.applymap(lambda x: float(x.replace(",", "")) if isinstance(x, str) else x)
</pre>

```python
processing_today_data = scale_data(processing_today_data)
```

<pre>
/home/dst78/anaconda3/envs/Deep/lib/python3.11/site-packages/sklearn/base.py:458: UserWarning: X has feature names, but StandardScaler was fitted without feature names
  warnings.warn(
</pre>

```python
processing_today_data
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
      <th>날짜</th>
      <th>변동성지수_종가</th>
      <th>변동성지수_시가</th>
      <th>변동성지수_고가</th>
      <th>변동성지수_저가</th>
      <th>변동성지수_변동 %</th>
      <th>미국10년국채_종가</th>
      <th>미국10년국채_시가</th>
      <th>미국10년국채_고가</th>
      <th>미국10년국채_저가</th>
      <th>...</th>
      <th>코스닥_종가</th>
      <th>코스닥_시가</th>
      <th>코스닥_고가</th>
      <th>코스닥_저가</th>
      <th>코스닥_변동 %</th>
      <th>다우존스_종가</th>
      <th>다우존스_시가</th>
      <th>다우존스_고가</th>
      <th>다우존스_저가</th>
      <th>다우존스_변동 %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2025-02-26</td>
      <td>-0.215555</td>
      <td>-0.279948</td>
      <td>-0.122741</td>
      <td>-0.187327</td>
      <td>0.251264</td>
      <td>1.115136</td>
      <td>1.186087</td>
      <td>1.151617</td>
      <td>1.141879</td>
      <td>...</td>
      <td>-0.577081</td>
      <td>-0.604755</td>
      <td>-0.61126</td>
      <td>-0.555613</td>
      <td>0.152306</td>
      <td>1.940698</td>
      <td>1.913117</td>
      <td>1.937721</td>
      <td>1.902119</td>
      <td>0.25768</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 36 columns</p>
</div>



```python
processing_today_data.set_index("날짜", inplace=True)
processing_today_data = np.array(processing_today_data)
```


```python
processing_today_data
```

<pre>
array([[-0.21555526, -0.27994777, -0.12274103, -0.18732721,  0.25126363,
         1.11513572,  1.18608685,  1.15161722,  1.14187937, -0.6523526 ,
         1.90130199,  1.96884296,  1.94014396,  1.88470489, -0.9071739 ,
         1.79514292,  1.77923581,  1.7189048 ,  1.82818019, -0.09019509,
         0.1469806 ,  0.10849754,  0.11955455,  0.13288753,  0.31966724,
        -0.57708119, -0.60475547, -0.61126023, -0.55561324,  0.15230583,
         1.94069812,  1.91311688,  1.9377209 ,  1.90211913,  0.25767998]])
</pre>

```python
with torch.no_grad():
    processing_today_data = processing_today_data[0:1]
    predict_data = torch.tensor(processing_today_data, dtype=torch.float32).unsqueeze(0)

    reconstructed_pred, latent_pred = autoencoder(predict_data)
    predicted_real = predictor(latent_pred)

    predicted_class_real = torch.argmax(predicted_real, dim=1).item()

    print("예측된 클래스:", predicted_class_real)
```

<pre>
예측된 클래스: 1
</pre>