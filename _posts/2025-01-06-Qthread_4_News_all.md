---
layout: single
title:  "[키움 자동매매 프로그램] - 시황 분석(News_all.py 및 Qthread_4.py) 구현"
categories: Kiwoom
tag: [python, coding, API]
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


메인 윈도우가 되는 pytrader.py에서 현재까지 계좌 정보, 계좌 관리, 자동매매를 구현했다.



이제는 각기 다른 윈도우로 이동하여 시황 분석, 세분화된 자동매매 페이지, 조건식 자동매매 페이지를 구현한다.



먼저 구현할 것은 시황 분석이다.



주식 시장의 중심이 되는 미국과 우리나라의 환율, 지수 등을 파악하여 투자하기에 적당한 시점인지를 파악한다.


> # News_all.py 구현


**pytrader.py**의 초기화 메서드 부분을 보면



```python

self.CRR.clicked.connect(self.Crolling)                     # 웹 크롤링

```



해당 부분과 **Crolling** 메서드를 구현한



```python

def Crolling(self) :

    print("뉴스 가져오기")

    self.second = secondwindow()

```



해당 부분을 보면 **secondWindow()** 클래스를 인스턴스화 한다.



그렇다면 **secondWindow()** 클래스를 구현해야 하는데, 이를 **News_all.py** 파일로 구현한다.



***


```python

import sys



from PyQt5.QtWidgets import *

from PyQt5 import uic



from Qthread_4 import Thread4       # 웹 크롤링



form_secondwindow = uic.loadUiType("pytrader2.ui")[0]



class secondwindow(QMainWindow, QWidget, form_secondwindow) :

    def __init__(self) :

        super(secondwindow, self).__init__()

        self.initUi()

        self.show()

        

        self.check_exchange_rate()

        

        self.pushButton.clicked.connect(self.btn_second_to_main)

        

    def initUi(self) :

        self.setupUi(self)

        

    def btn_second_to_main(self) :      # 닫기 버튼

        self.close()

        

    def check_exchange_rate(self) :

        print("환율 가져오기")

        h4 = Thread4(self)

        h4.start()

        

if __name__ == "__main__" :

    app = QApplication(sys.argv)

    CH = secondwindow()

    CH.show()

    app.exec_()

```



해당 클래스에서는 윈도우를 조작하는 기능만 구현한다.



해당 윈도우를 열면 자동으로 시황을 분석하여 값들이 나오는데 이는 Thread4에서 구현한다.



**닫기** 버튼을 누르면 윈도우가 닫히는 기능만 구현하였다.



***


> # Qthread_4.py 구현


먼저 화면을 보면 다음과 같다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/시황_분석_페이지.png?raw=true)



환율, 서부텍사스유, 미국 국채 10년 등등 여러가지 지수 상황들이 나와있는데 이는 **웹 크롤링**으로 데이터를 받아와 구현할 수 있다.


>> ## Qthread_4.py - init 생성


```python

from PyQt5.QtCore import *

from kiwoom import Kiwoom

from urllib.request import urlopen

from bs4 import BeautifulSoup



class Thread4(QThread) :

    def __init__(self, parent) :

        super().__init__(parent)

        self.parent = parent

        

        self.k = Kiwoom()

        

        ### 웹크롤링 정보 가져오기

        self.US_Exchange_Rate()     # 달러 환율

        self.West_Texas_Oil()       # 서부 텍사스유

        self.Treasury_Bond()        # 미국 국채 10년

        self.CPI()                  # 한국 소비자 물가 지수

        self.VIX()                  # 변동성 지수

        self.ADR()                  # ADR 지표

        self.DowJones()             # 다우 존스

        self.Nasdaq()               # 나스닥

        self.Cospi()                # 코스피

        self.Cosdaq()               # 코스닥

        ###

```



각각의 지수들을 메서드로 구현한다.



***


>> ## 시황 분석 - 웹 크롤링으로 원하는 데이터 가져오기


지수들을 분석하기 위해서 웹 크롤링을 어떤 방식으로 하는지 알아야 한다.



먼저 원하는 페이지를 들어간다. 여기서는 달러이므로 네이버에 간단하게 환율로 치면 달러 환율이 나온다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/환율.PNG?raw=true)



해당 페이지의 url 주소를 복사해준 다음 라이브러리를 이용하여 웹 페이지의 정보들을 가져올 수 있다.



예를 들면



```python
from urllib.request import urlopen
from bs4 import BeautifulSoup

response = urlopen("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%ED%99%98%EC%9C%A8")
soup = BeautifulSoup(response, "html.parser")
```

**response** 변수는 해당 URL에 접속하여 HTML 페이지의 내용을 가져온다.



**soup** 변수는 해당 HTML 문서를 트리 형태로 변환한 객체로 바꿔주는데 이를 통해 특정 태그, 클래스 등을 조작할 수 있게 해준다.



해당 결과로는 다음과 같이 웹 HTML을 긁어온다.(길이가 길기에 뒷 부분 생략)



```python
length = 300

print(str(soup)[:length])
```

<pre>
<!DOCTYPE html>
 <html lang="ko"><head> <meta charset="utf-8"/> <meta content="always" name="referrer"/> <meta content="telephone=no,address=no,email=no" name="format-detection"/> <meta content="환율 : 네이버 검색" property="og:title"> <meta content="https://ssl.pstatic.net/sstatic/search/common/og_v3.png"
</pre>
우리가 현재 **환율** 페이지에서 얻고 싶은 정보는 환율 가격, 전일 대비 증가 가격, 전일 대비 증가율이다.(환율 기준)



웹 페이지에서 **F12**를 누르고 **Ctrl+F** 로 원하는 정보값을 검색한다.



위의 환율 페이지에서 1달러당 1,465.30 달러에 거래되고 있으므로 **1,465.30**으로 검색해보면 된다.



그러면 다음과 같이 값에 해당하는 부분이 나오고, 실제 페이지에서 어떤 부분에 해당하는지 파란색으로 알려준다.



여기서 필요한 것이 **은행 고시환율**에 해당하는 부분을 크롤링 해야한다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/1_465.PNG?raw=true)



원하는 부분의 데이터를 보면 class는 **spt_con dw**로 되어있다.



또한 **spt_con dw** 내부의 클래스인 **n_ch** 클래스를 확인하면 하락인지 증가했는지, 얼마나 증감했는지, 증감율이 나온다.



이 부분을 추출하려면 다음과 같이 코드를 작성하면 된다.



```python
value = soup.find("span", {"class" : "spt_con dw"})

value = value.text.split()

a = value[0][0:10]
b = value[1][4:6]
c = value[1][6:11]
d = value[2][0:7]

print(a, b, c, d)
```

<pre>
1,465.30 하락 5.70 -0.39%
</pre>
여기서 유의할 점은 **span class**는 두 종류이다.



하락이면 **spt_con dw**이고 상승이면 **spt_con up**으로 나타난다.



따라서 해당 조건에 맞춰서 코드를 작성하면 다음과 같이 작성할 수 있다.



```python

def US_Exchange_Rate(self) :

    response = urlopen("https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%ED%99%98%EC%9C%A8")

    soup = BeautifulSoup(response, "html.parser")

    

    value = soup.find("span", {"class" : "spt_con dw"})

    

    if value is not None :

        value2 = value.text.split()

    else :

        value = soup.find("span", {"class" : "spt_con up"})

        value2 = value.text.split()

        

    print(value2)

    

    a = value2[0][0:10]         # 환율 가격

    b = value2[1][4:6]          # 전일 대비 증감 확인

    c = value2[1][6:11]         # 전일 대비 증감 가격

    d = value2[2][0:7]          # 전일 대비 증가율

    

    self.parent.exchange_1.setPlainText(a)

    self.parent.exchange_2.setPlainText(b)

    self.parent.exchange_3.setPlainText(c)

    self.parent.exchange_4.setPlainText(d)

    

    self.parent.exchange_1.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.parent.exchange_2.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.parent.exchange_3.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.parent.exchange_4.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    

    d = d.rstrip('%')

    

    if "-" in d and float(d[1:5]) > 1 :                 # 환율 하락, 하락 폭 1% 이상

        self.parent.exchange_5.setPlainText(str(100))

    elif "-" in d and float(d[1:5]) < 1 :               # 환율 하락, 하락 폭 1% 이하

        self.parent.exchange_5.setPlainText(str(75))

    elif "-" not in d and float(d[1:5]) < 1 :           # 환율 상승, 상승 폭 1% 이하

        self.parent.exchange_5.setPlainText(str(50))

    elif "-" not in d and float(d[1:5]) > 1 :           # 환율 상승, 상승 폭 1% 이상

        self.parent.exchange_5.setPlainText(str(25))

        

    self.parent.exchange_5.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

```



웹 크롤링을 통해 원하는 데이터를 얻어오고 UI에 전시해준다.



또한, 얻어온 데이터를 바탕으로 투자 점수를 매기기위한 기본적인 알고리즘을 작성한다.



이러한 방식으로 자신이 원하는 지수(필자는 앞 단락에서의 지수들을 기입)를 웹 크롤링하면 다음과 같이 전시할 수 있다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/전체_시황.PNG?raw=true)



***


투자점수 확인은 차후 여러 지수 데이터들을 이용하여 머신러닝으로 학습을 시도해 보려고 한다.

