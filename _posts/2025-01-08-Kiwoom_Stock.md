---
layout: single
title:  "[키움 자동매매 프로그램] - 조건식 자동매매 페이지(Kiwoom_Stock.py) 구현"
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


전체 흐름의 마지막인 **조건식**을 이용한 자동매매 구현만이 남았다.



이전 자동매매에서는 사용자가 직접 종목을 선정하여 그 종목에 대해 익절, 손절 분기점을 설정하여 자동매매를 진행하였으나



이번에는 키움에서 제공하는 조건식들을 이용하여 자동으로 종목을 선정하고 해당 종목을 매매하는 알고리즘을 구현한다.


> # Kiwoom_Stock.py


먼저 우리가 구현할 윈도우를 보면 다음과 같다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/조건식_자동매매_페이지.png?raw=true)


**조건식 검색**, **자동매매 파라미터 입력**, **실시간 자동매매** 세 박스를 구분하여 하나씩 **Thread**로 구현할 것이다.



구현 전 **조건식 자동매매** 페이지의 UI를 코드와 연결해야한다.


>> ## Kiwoom_Stock - init() 생성


```python

from PyQt5.QtWidgets import *

from PyQt5 import uic

from PyQt5.QtCore import *



from kiwoom import Kiwoom



import os



from Qthread_7 import Thread7       # 조건검색식 목록 가져오기

from Qthread_8 import Thread8       # 조건검색식 종목 가져오기

from Qthread_9 import Thread9       # 조건검색식 자동매매 시작하기

from Qthread_10 import Thread10     # 조건검색식 종목 자동으로 가져오기



form_forthwindow = uic.loadUiType("pytrader4.ui")[0]



class Forthwindow(QMainWindow, QWidget, form_forthwindow) :

    def __init__(self) :

        super(Forthwindow, self).__init__()

        self.initUi()

        self.show()

        

        self.k = Kiwoom()

        

        self.get_kiwoom_list()      # 조건검색식 목록 가져오기

        

        self.search_kiwoom_stock.clicked.connect(self.search_stock_nauto)     # 조건검색식 종목 가져오기(수동)

        self.auto_search.clicked.connect(self.search_stock_auto)              # 조건검색식 종목 가져오기(자동)

        

        self.Start_Auto.clicked.connect(self.Save_selected_code)

        self.Del_Stock.clicked.connect(self.delete_code)

        self.Load_Stock.clicked.connect(self.Load_code)

        

        self.Start_Auto.clicked.connect(self.start_stock)

        

        ### 자동검색 간격 설정

        self.search_minute.setValue(2)

        self.search_minute.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.search_minute.setDecimals(0)

        ###

        

        ### 매수 관련 초기화

        self.price_stock.setValue(1000000)

        self.price_stock.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.price_stock.setDecimals(0)

        ###

        

        ### 매수 파라미터 초기화

        self.text_edit1.setValue(0)

        self.text_edit2.setValue(25)

        self.text_edit3.setValue(-1)

        self.text_edit4.setValue(25)

        self.text_edit5.setValue(-2)

        self.text_edit6.setValue(25)

        self.text_edit7.setValue(-3)

        self.text_edit8.setValue(25)

        ###

        

        ### 익절 파라미터 초기화

        self.text_edit9.setValue(3)

        self.text_edit10.setValue(50)

        self.text_edit11.setValue(5)

        self.text_edit12.setValue(50)

        ###

        

        ### 손절 파라미터 초기화

        self.text_edit13.setValue(-5)

        self.text_edit14.setValue(50)

        self.text_edit15.setValue(-10)

        self.text_edit16.setValue(50)

        ###

        

        ### 시간 정보 초기화

        self.stop_time.setDisplayFormat('hh:mm:ss')

        self.start_time.setDisplayFormat('hh:mm:ss')

        self.stop_time.setTime(QTime(14, 00,  00))

        self.start_time.setTime(QTime(14, 10, 00))

        ###

        

        ### 더블 스핀 박스 우측정렬, 소수점 삭제

        self.text_edit1.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit1.setDecimals(0)

        self.text_edit2.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit2.setDecimals(0)

        self.text_edit3.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit3.setDecimals(0)

        self.text_edit4.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit4.setDecimals(0)

        self.text_edit5.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit5.setDecimals(0)

        self.text_edit6.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit6.setDecimals(0)

        self.text_edit7.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit7.setDecimals(0)

        self.text_edit8.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit8.setDecimals(0)

        self.text_edit9.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit9.setDecimals(0)

        self.text_edit10.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit10.setDecimals(0)

        self.text_edit11.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit11.setDecimals(0)

        self.text_edit12.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit12.setDecimals(0)

        self.text_edit13.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit13.setDecimals(0)

        self.text_edit14.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit14.setDecimals(0)

        self.text_edit15.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit15.setDecimals(0)

        self.text_edit16.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.text_edit16.setDecimals(0)

        ###



    def initUi(self) :

        self.setupUi(self)

```



UI에 전시하였던 박스, 버튼 등을 코드와 연결하고, 초기화 작업을 수행한다.



***


>> ## 데이터베이스 조작


조건검색식을 알아보기 전 데이터베이스에 저장하고 불러오기 및 삭제 코드를 구현한다.



이전에 구현하였던 것과 유사하다.


>>> ### 데이터베이스 - Load_code


```python

def Load_code(self) :

    if os.path.exists("dist/Kiwoom_Parameter.txt") :

        f = open("dist/Kiwoom_Parameter.txt", "r", encoding="utf8")

        lines = f.readlines()

        

        for line in lines :

            if line != "" :

                ls = line.split("\t")

                re_time = ls[0]

                st_price = ls[1]

                a1 = ls[2]

                a2 = ls[3]

                a3 = ls[4]

                a4 = ls[5]

                a5 = ls[6]

                a6 = ls[7]

                a7 = ls[8]

                a8 = ls[9]

                a9 = ls[10]

                a10 = ls[11]

                a11 = ls[12]

                a12 = ls[13]

                a13 = ls[14]

                a14 = ls[15]

                a15 = ls[16]

                a16 = ls[17].split("\n")[0]

                

                self.Getanal_code = [re_time, st_price, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16]

        f.close()

        

    self.search_minute.setValue(float(self.Getanal_code[0]))

    self.price_stock.setValue(float(self.Getanal_code[1]))

    self.text_edit1.setValue(float(self.Getanal_code[2]))

    self.text_edit2.setValue(float(self.Getanal_code[3]))

    self.text_edit3.setValue(float(self.Getanal_code[4]))

    self.text_edit4.setValue(float(self.Getanal_code[5]))

    self.text_edit5.setValue(float(self.Getanal_code[6]))

    self.text_edit6.setValue(float(self.Getanal_code[7]))

    self.text_edit7.setValue(float(self.Getanal_code[8]))

    self.text_edit8.setValue(float(self.Getanal_code[9]))

    self.text_edit9.setValue(float(self.Getanal_code[10]))

    self.text_edit10.setValue(float(self.Getanal_code[11]))

    self.text_edit11.setValue(float(self.Getanal_code[12]))

    self.text_edit12.setValue(float(self.Getanal_code[13]))

    self.text_edit13.setValue(float(self.Getanal_code[14]))

    self.text_edit14.setValue(float(self.Getanal_code[15]))

    self.text_edit15.setValue(float(self.Getanal_code[16]))

    self.text_edit16.setValue(float(self.Getanal_code[17]))

    

    self.search_minute.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.search_minute.setDecimals(0)

    self.price_stock.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.price_stock.setDecimals(0)

    self.text_edit1.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit1.setDecimals(0)

    self.text_edit2.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit2.setDecimals(0)

    self.text_edit3.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit3.setDecimals(0)

    self.text_edit4.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit4.setDecimals(0)

    self.text_edit5.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit5.setDecimals(0)

    self.text_edit6.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit6.setDecimals(0)

    self.text_edit7.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit7.setDecimals(0)

    self.text_edit8.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit8.setDecimals(0)

    self.text_edit9.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit9.setDecimals(0)

    self.text_edit10.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit10.setDecimals(0)

    self.text_edit11.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit11.setDecimals(0)

    self.text_edit12.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit12.setDecimals(0)

    self.text_edit13.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit13.setDecimals(0)

    self.text_edit14.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit14.setDecimals(0)

    self.text_edit15.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit15.setDecimals(0)

    self.text_edit16.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

    self.text_edit16.setDecimals(0)

```



이전에 데이터베이스에 저장된 데이터를 불러오는 것과 거의 유사하다.



다른점으로는 이전에는 데이터베이스에서 데이터를 불러온 다음 **buylast** 라는 하나의 큰 박스에 전시하기에 그대로 전시하면 됬었다.



하지만 우리가 이번 조건검색식 자동매매를 할 때에는 파라미터들(매수 파라미터, 익절 파라미터, 손절 파라미터 등)을 불러오고 각 **spinbox** 등에 넣기 때문에 Load함과 동시에 UI에 전시해주는 부분을 직접 구현해야 한다.


>>> ### 데이터베이스 - Save_selected_code


```python

def Save_selected_code(self) :

    re_time = self.search_minute.value()

    st_price = self.price_stock.value()



    a1 = self.text_edit1.value()

    a2 = self.text_edit2.value()

    a3 = self.text_edit3.value()

    a4 = self.text_edit4.value()

    a5 = self.text_edit5.value()

    a6 = self.text_edit6.value()

    a7 = self.text_edit7.value()

    a8 = self.text_edit8.value()

    a9 = self.text_edit9.value()

    a10 = self.text_edit10.value()

    a11 = self.text_edit11.value()

    a12 = self.text_edit12.value()

    a13 = self.text_edit13.value()

    a14 = self.text_edit14.value()

    a15 = self.text_edit15.value()

    a16 = self.text_edit16.value()

    

    f = open("dist/Kiwoom_Parameter.txt", "a", encoding="utf8")

    f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % 

            (re_time, st_price, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16))

    f.close()

```


>>> ### 데이버베이스 - delete_code


```python

def delete_code(self) :

    if os.path.exists("dist/Kiwoom_Parameter.txt") :

        os.remove("dist/Kiwoom_Parameter.txt")

```


>> ## 조건검색식 설정하기


프로그램에 조건검색식을 적용하기 위해서는 먼저 조건검색식을 만들어야 한다.



키움 OpenAPI는 **영웅문4**라는 PC용 키움 거래 애플리케이션의 데이터를 사용하기 때문에 영웅문4의 설치가 필요하다.


일단 영웅문4를 열게되면 처음으로 보는 창은 다음과 같다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/영웅문4.png?raw=true)



***



여기서 우리가 볼 부분은 **조건 검색**으로 나와있는 툴바이다.



해당 부분을 클릭하고 대상변경 → 코스피/코스닥 선택 → 제외종목은 개인 투자 성향별로 선택 후 확인 버튼을 누른다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/조건검색_클릭.png?raw=true)



***



이제 왼쪽 툴바를 보면 **조건식**, **추천식**, **내조건식** 이 3가지가 나오는데, 추천식을 사용해도 되고, 조건식에서 사용자의 취향대로 조건을 설정하면 된다.(자세한 조건검색식 사용 방법은 여러 블로그 및 인터넷에 나와있다.)



예를 들면, **눌림목 이후 상승 포착**의 조건식을 만들고 저장, 만든 조건식으로 검색 버튼을 누르면 해당하는 종목들이 나오게 된다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/조건식_예시.png?raw=true)



***



위의 이미지에서 빨간 동그라미 부분을 수정함으로써 조건검색식의 기준 값들을 설정할 수 있다.


>> ## 조건검색식 목록, 종목 가져오기


영웅문4에서 조건검색식 생성을 마쳤다면 이제 프로그램에 조건검색식을 불러오고 해당 조건에 맞는 종목을 가져와 거래를 해야한다.



해당 부분은 각 Thread로 구현하며 Kiwoom_Stock.py 코드 내 메서드는 다음과 같이 정의한다.



```python

def get_kiwoom_list(self) :

    print("조건검색식 목록 가져오기")

    h7 = Thread7(self)

    h7.start()

    

def search_stock_nauto(self) :

    print("조건검색식 종목 가져오기(수동)")

    h8 = Thread8(self)

    h8.start()



def start_stock(self) :

    print("조건검색식 자동매매 시작")

    h9 = Thread9(self)

    h9.start()

    

def search_stock_auto(self) :

    print("조건검색식 종목 가져오기(자동)")

    h10 = Thread10(self)

    h10.start()

```



***

