---
layout: single
title:  "AWS를 활용한 LSTM 모델 예측 시스템 구축"
categories: 
tag: [python, AWS, coding, Machine Learning]
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


지난 포스팅에서는 시황 분석 모델을 훈련시켰고, 해당 모델을 키움 OpenAPI를 이용하여 만들었던 프로그램에 적용한다.



여기서 한 가지 문제를 직면하게 되는데, 모델을 훈련시켰던 환경은 64비트의 ubuntu 환경이고



키움 OpenAPI는 32비트의 환경에서만 돌아가기 때문에 훈련시켰던 모델을 불러와서 사용할 수 없었다.



따라서 aws를 이용하여 모델과 코드 파일을 업로드하고 aws 서버에서 모델의 예측값을 받아오는 시스템을 구현한다.



***


# AWS 모델, 코드 파일 업로드


가장 먼저 EC2에 인스턴스를 하나 생성하고 키, 보안 그룹을 설정한다.



인스턴스의 볼륨 크기는 32 GiB로 torch 라이브러리의 크기를 고려하여 넉넉하게 잡아주었고, 인스턴스 유형은 c5.large로 해주었다.



보안그룹의 경우 아래 이미지와 같이 HTTP, SSH, HTTPS와 실제 모델 예측을 받아올 때 사용할 5000번 포트까지 설정해주었다.(443으로 접속하려 했으나 명령어에 sudo를 붙여줘야하는 불편함이 매번 있어 1000번 이후대로 포트 번호를 따로 지정해주었다.)


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/보안그룹.png?raw=true)



***



훈련시켰던 LSTM 모델과 전처리를 위한 스케일러를 S3에 업로드한다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/S3.png?raw=true)



***


이제 EC2 상으로 올릴 코드 파일을 업로드해준다.



.ssh 폴더에는 키 파일과 LSTM모델을 정의하고 데이터를 전처리하는 파일, 실제 코드를 수행하는 파일이 존재한다.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/ssh폴더.png?raw=true)



***


## real_time_model.py (LSTM 모델 정의 및 데이터 전처리)


LSTM모델을 정의하고 데이터를 전처리하는 파일인 real_time_model.py는 다음과 같다.



```python

import pandas as pd

import numpy as np

import re

import itertools

import os

import glob

import time

import joblib

import requests



import torch

import torch.nn as nn

import torch.optim as optim



from datetime import datetime

from bs4 import BeautifulSoup



from sklearn.preprocessing import StandardScaler



# LSTM AutoEncoder 모델 정의

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



# 미래 예측 모델 정의    

class FuturePredictor(nn.Module) :

    def __init__(self, latent_dim, prediction_dim) :

        super(FuturePredictor, self).__init__()

        self.lstm = nn.LSTM(latent_dim, 64, batch_first=True)

        self.fc = nn.Linear(64, prediction_dim)

        

    def forward(self, z) :

        lstm_out, _ = self.lstm(z)

        future_values = self.fc(lstm_out[:, -1, :])

        return future_values



# 모델 로드 및 예측    

class StockPredictor:

    def __init__(self, input_dim=35, hidden_dim=64, latent_dim=35, num_classes=4):

        self.autoencoder = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)

        self.predictor = FuturePredictor(latent_dim, num_classes)

        

        self.autoencoder.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "autoencoder.pt"), weights_only=True))

        self.predictor.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "predictor.pt"), weights_only=True))



        self.autoencoder.eval()

        self.predictor.eval()



    def predict(self, data) :

        with torch.no_grad():

            data = data[0:1]

            predict_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)



            reconstructed_pred, latent_pred = self.autoencoder(predict_data)

            predicted_real = self.predictor(latent_pred)



            predicted_class_real = torch.argmax(predicted_real, dim=1).item()



        return predicted_class_real



# 실시간 시황 데이터 가져오기        

class DataScraper :

    def __init__(self) :

        self.headers = {

            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

        }



    def get_data(self, url, tag, class_name):

        try:

            response = requests.get(url, headers=self.headers)

            response.raise_for_status()

            

            soup = BeautifulSoup(response.text, "html.parser")

            

            element = soup.find(tag, class_=class_name)

            if element:

                real_time_data = [col.get_text(strip=True) for col in element]

            else:

                real_time_data = "데이터를 찾을 수 없음"



        except Exception as e:

            real_time_data = f"데이터 불러오기 실패: {e}"

        

        return real_time_data



# 실시간 데이터 처리 및 저장

class DataProcessor :

    def __init__(self) :

        self.page_names = ["VIX", "US10", "Nasdaq", "Exchange", "Kospi", "Kosdaq", "DowJones"]

        self.urls = {

            "VIX": "https://www.investing.com/indices/volatility-s-p-500-historical-data",

            "US10": "https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data",

            "Nasdaq": "https://www.investing.com/indices/nasdaq-composite-historical-data",

            "Exchange": "https://www.investing.com/currencies/usd-krw-historical-data",

            "Kospi": "https://www.investing.com/indices/kospi-historical-data",

            "Kosdaq": "https://www.investing.com/indices/kosdaq-historical-data",

            "DowJones": "https://www.investing.com/indices/us-30-historical-data"

        }

        self.tag = "tr"

        self.class_name = "historical-data-v2_price__atUfP"

        self.inf_list = ["종가", "시가", "고가", "저가", "변동 %"]

        self.col_list = ["변동성지수", "미국10년국채", "나스닥", "환율", "코스피", "코스닥", "다우존스"]



    def data_preprocessing(self, data, stock_name) :

        if len(data) > 6 :

            return [stock_name] + [data[i] for i in [1, 2, 3, 4, 6]]

        else :

            return [stock_name] + [data[i] for i in [1, 2, 3, 4, 5]]



    def save_data(self, scraper):

        results= {}

        for key, url in self.urls.items() :

            results[key] = scraper.get_data(url, self.tag, self.class_name)



        today_data = [self.data_preprocessing(result, key) for key, result in results.items()]

        today = datetime.today().strftime('%Y-%m-%d')



        column_names = ["날짜"] + [f"{col}_{inf}" for col in self.col_list for inf in self.inf_list]

        flattened_data = [today] + list(itertools.chain.from_iterable(row[1:] for row in today_data))



        df = pd.DataFrame([flattened_data], columns=column_names)

        df.to_csv(f"{today}_trdata.csv", index=False)

        

# 모델 예측에 사용할 수 있도록 저장된 데이터를 불러온 다음 전처리

class RealDataPreprocessing :

    def __init__(self, scaler_path="scaler.pkl") :

        script_dir = os.path.dirname(os.path.abspath(__file__))

        

        if not os.path.isabs(scaler_path):

            scaler_path = os.path.join(script_dir, scaler_path)



        self.scaler_path = scaler_path

        self.scaler = joblib.load(scaler_path)

        

    def convert_percent(self, x) :

            if isinstance(x, str) and "%" in x :

                return float(x.replace("%", "")) / 100.0

            return x



    def merge_preprocessing(self, dataset_list) :

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

        standard_data.ffill(inplace=True)

        

        # 데이터 날짜 기준 정렬

        standard_data = standard_data.sort_values(by="날짜", ascending=True).reset_index(drop=True)

        

        # %로 표현된 값들 수치형 데이터로 표현

        percent_columns = [col for col in standard_data.columns if col.endswith("_변동 %")]

        standard_data[percent_columns] = standard_data[percent_columns].map(self.convert_percent)

        

        # ","가 포함된 object 타입 수치형 데이터로 변환

        standard_data = standard_data.map(lambda x: float(x.replace(",", "")) if isinstance(x, str) else x)

        

        return standard_data



    def scale_data(self, dataset, scaler_path="scaler.pkl"):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        if not os.path.isabs(scaler_path):

            scaler_path = os.path.join(script_dir, scaler_path)

            

        scaler = joblib.load(scaler_path)



        numeric_data = dataset.select_dtypes(include=[np.number])

        scaled_data = scaler.transform(numeric_data)

        dataset[numeric_data.columns] = scaled_data

        

        return dataset

    

    def process_real_data(self, file_name) :

        today_data = pd.read_csv(file_name)

        today_data["날짜"] = pd.to_datetime(today_data["날짜"])



        processing_today_data = self.merge_preprocessing(today_data)

        processing_today_data = self.scale_data(processing_today_data)



        processing_today_data.set_index("날짜", inplace=True)

        return np.array(processing_today_data)

    

input_dim = 35

hidden_dim = 64

latent_dim = 35

num_classes = 4    



autoencoder = LSTMAutoencoder(input_dim, hidden_dim, latent_dim)

predictor = FuturePredictor(latent_dim, num_classes)



autoencoder.eval()

predictor.eval()

```



***


## predict.py (실제 코드를 수행)


**flask**를 이용하여 모델 예측을 수행하는 API 서버를 설정하였다.



```python

import real_time_model as rtm

from datetime import datetime

from flask import Flask, jsonify



app = Flask(__name__)



@app.route("/predict", methods=["GET"])



def run_predict() :

    try :

        scraper = rtm.DataScraper()

        processor = rtm.DataProcessor()

        processor.save_data(scraper)



        predictor = rtm.StockPredictor()



        real_time_processor = rtm.RealDataPreprocessing()

        today = datetime.today().strftime('%Y-%m-%d')

        processed_real_data = real_time_processor.process_real_data(f"{today}_trdata.csv")



        prediction_result = predictor.predict(processed_real_data)



        return jsonify(prediction_result)



    except Exception as e :

        return jsonify({"error" : str(e)}), 500

```



***


위처럼 생성된 두 파일을 **scp** 명령어를 이용하여 EC2에 업로드해준다.





```bash

scp -i Market_analysis.model.pem real_time_model.py ubuntu@ec2-52-79-99-15.ap-northeast-2.compute.amazonaws.com:/home/ubuntu



scp -i Market_analysis.model.pem predict.py ubuntu@ec2-52-79-99-15.ap-northeast-2.compute.amazonaws.com:/home/ubuntu

```



또한 S3 버킷에 업로드하였던 모델 파일과 스케일러 파일도 EC2에 복사해준다.



```bash

aws configure



aws s3 cp s3://jmstore/autoencoder.pt /home/ubuntu/autoencoder.pt

aws s3 cp s3://jmstore/predictor.pt /home/ubuntu/predictor.pt

aws s3 cp s3://jmstore/scaler.pkl /home/ubuntu/scaler.pkl

```



이제 인스턴스를 실행하고 ssh 명령어를 이용해 EC2에 접속해보면 다음과 같이 파일이 업로드 된 것을 볼 수 있다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/업로드확인.png?raw=true)



***


# 모델 예측 처리


Flask를 사용하여 작성된 **predict.py** 코드를 실행한다.



해당 코드 파일은 백그라운드에서 서버를 실행시켜 요청을 받으면 응답으로 예측값을 돌려준다.



```bash

nohup /home/ubuntu/myenv/bin/python3 /home/ubuntu/predict.py > server.log 2>&1 &

```



해당 명령어로 백그라운드 실행 후 확인해 보면 다음과 같이 프로세스가 실행 중임을 확인할 수 있다.


![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/프로세스_실행_확인.png?raw=true)



***


이제 로컬에서 요청을 보내 예측값을 받아오는 코드를 작성후 테스트를 해본다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/응답_테스트.png?raw=true)




이상없이 예측값을 잘 받아오는 것을 볼 수 있다.



다음 포스팅에서는 자동매매 프로그램에 시황 분석 페이지에 실제로 적용해 보는 과정을 진행한다.

