---
layout: single
title:  "[키움 자동매매 프로그램] - 분할 자동매매 페이지(Division_Stock.py) 구현"
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


다음으로 구현할 부분은 **분할 자동매매**이다.



메인 윈도우의 빨간 동그라미 부분에 해당한다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/분할_매수매도.PNG?raw=true)



***



먼저 **분할 매수매도**의 윈도우의 UI와 코드를 연결하고 거래를 제외한 나머지 기능들을 구현하기 위한 Division_Stock.py를 생성한다.


> # Division_Stock.py 구현


먼저 UI를 다시 살펴보면



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/자동매매_페이지.png?raw=true)



거래할 종목명을 메인 윈도우에서 지정한 다음 해당 페이지에서 입력 후 종목 추가(또는 종목코드 추가), 감시가격 설정, DB에 저장, 자동매매 시작 및 종료 기능을 구현해야한다.


>> ## Division_Stock - init() 생성


```python

import sys

import os



from PyQt5.QtWidgets import *

from PyQt5 import uic

from PyQt5.QtCore import *



from kiwoom import Kiwoom



from Qthread_6 import Thread6       # 분할 자동 매매



form_thirdwindow = uic.loadUiType("pytrader3.ui")[0]



class Thirdwindow(QMainWindow, QWidget, form_thirdwindow) :

    def __init__(self) :

        super(Thirdwindow, self).__init__()

        self.initUi()

        self.show()

        

        self.doubleSpinBox_1.setValue(0)

        self.doubleSpinBox_2.setValue(0)

        self.doubleSpinBox_3.setValue(0)

        self.doubleSpinBox_4.setValue(0)

        self.doubleSpinBox_5.setValue(0)

        self.doubleSpinBox_6.setValue(0)

        self.doubleSpinBox_7.setValue(0)

        self.doubleSpinBox_8.setValue(0)

        self.doubleSpinBox_9.setValue(0)

        self.doubleSpinBox_10.setValue(0)

        self.doubleSpinBox_11.setValue(0)

        self.doubleSpinBox_12.setValue(0)

        self.doubleSpinBox_13.setValue(0)

        self.doubleSpinBox_14.setValue(0)

        self.doubleSpinBox_15.setValue(0)

        self.doubleSpinBox_16.setValue(0)

        

        self.doubleSpinBox_1.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_1.setDecimals(0)

        self.doubleSpinBox_2.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_2.setDecimals(0)

        self.doubleSpinBox_3.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_3.setDecimals(0)

        self.doubleSpinBox_4.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_4.setDecimals(0)

        self.doubleSpinBox_5.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_5.setDecimals(0)

        self.doubleSpinBox_6.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_6.setDecimals(0)

        self.doubleSpinBox_7.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_7.setDecimals(0)

        self.doubleSpinBox_8.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_8.setDecimals(0)

        self.doubleSpinBox_9.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_9.setDecimals(0)

        self.doubleSpinBox_10.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_10.setDecimals(0)

        self.doubleSpinBox_11.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_11.setDecimals(0)

        self.doubleSpinBox_12.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_12.setDecimals(0)

        self.doubleSpinBox_13.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_13.setDecimals(0)

        self.doubleSpinBox_14.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_14.setDecimals(0)

        self.doubleSpinBox_15.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_15.setDecimals(0)

        self.doubleSpinBox_16.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        self.doubleSpinBox_16.setDecimals(0)

        

        self.k = Kiwoom()

        

        self.detail_account_info_event_loop = QEventLoop()

        

        self.k.kiwoom.OnReceiveTrData.connect(self.trdata_slot)

        self.additemlast.clicked.connect(self.searchItem2)

        

        self.Deletecode.clicked.connect(self.deletecode)

        

        self.Getanal_code = []

        self.Load_Stock.clicked.connect(self.Load_code)

        self.Save_Stock.clicked.connect(self.Save_selected_code)

        self.Del_Stock.clicked.connect(self.delete_code)

        

        self.Start_Auto.clicked.connect(self.start_real_auto)

        self.Stop_Everyting.clicked.connect(self.stop_auto)



    def initUi(self) :

        self.setupUi(self)

```



UI를 불러오고 **doubleSpinBox**는 매수/매도 가격 및 수량을 설정할 수 있는 박스이다.


>> ## Divison_Stock - 서버 데이터 요청 및 수신


```python

self.k.kiwoom.OnReceiveTrData.connect(self.trdata_slot)

self.additemlast.clicked.connect(self.searchItem2)

```



메인 윈도우에 자동매매를 구현했던 것과 마찬가지로 먼저 거래할 종목에 대한 데이터를 받아오고 추가해야한다.



***


```python

def trdata_slot(self, sCrNo, sRQName, sTrCode, sRecordName, sPrevNext) :

    if sTrCode == "opt10001" :

        if sRQName == "주식기본정보요청" :

            currentPrice = abs(int(self.k.kiwoom.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, "현재가")))

            row_count = self.buylast.rowCount()

            

            self.buylast.setItem(row_count - 1, 2, QTableWidgetItem(str(format(int(currentPrice), ","))))

            

        self.detail_account_info_event_loop.exit()

```



이전의 ppytrader.py에 구현했던 것과 유사하게 구현하되, 매수 가격, 매수 수량 등의 파라미터들은 감시박스에서 구현하므로



거래 종목의 **현재가**만 **buylast**에 기입한다.



***


```python

def searchItem2(self) :

    self.itemName = self.searchItemTextEdit2.toPlainText()

    self.new_code = None

    

    if self.itemName != "" :

        for code in self.k.All_Stock_Code.keys() :

            if self.itemName == self.k.All_Stock_Code[code]['종목명'] :

                self.new_code = code

                

    if self.new_code != "" and self.itemName == "" :

        self.new_code = self.searchItemTextEdit3.toPlainText().strip()

        

        for code in self.k.All_Stock_Code.keys() :

            if self.new_code == code :

                self.itemName = self.k.All_Stock_Code[code]['종목명']

                

    column_head = ["종목코드", "종목명", "현재가", "매수가격_1", "매수수량_1", "매수가격_2", "매수수량_2", "매수가격_3", "매수수량_3", "매수가격_4", "매수수량_4",

                    "매도가격_1", "매도수량_1", "매도가격_2", "매도수량_2", "매도가격_3", "매도수량_3", "매도가격_4", "매도수량_4"]

    colCount = len(column_head)

    row_count = self.buylast.rowCount()

    

    self.buylast.setColumnCount(colCount)

    self.buylast.setRowCount(row_count+1)

    self.buylast.setHorizontalHeaderLabels(column_head)

    

    self.buylast.setItem(row_count, 0, QTableWidgetItem(str(self.new_code)))

    self.buylast.setItem(row_count, 1, QTableWidgetItem(str(self.itemName)))

    self.getItemInfo(self.new_code)

    

    self.buylast.setItem(row_count, 3, QTableWidgetItem(str(format(int(self.doubleSpinBox_1.value()), ","))))

    self.buylast.setItem(row_count, 4, QTableWidgetItem(str(format(int(self.doubleSpinBox_2.value()), ","))))

    self.buylast.setItem(row_count, 5, QTableWidgetItem(str(format(int(self.doubleSpinBox_3.value()), ","))))

    self.buylast.setItem(row_count, 6, QTableWidgetItem(str(format(int(self.doubleSpinBox_4.value()), ","))))

    self.buylast.setItem(row_count, 7, QTableWidgetItem(str(format(int(self.doubleSpinBox_5.value()), ","))))

    self.buylast.setItem(row_count, 8, QTableWidgetItem(str(format(int(self.doubleSpinBox_6.value()), ","))))

    self.buylast.setItem(row_count, 9, QTableWidgetItem(str(format(int(self.doubleSpinBox_7.value()), ","))))

    self.buylast.setItem(row_count, 10, QTableWidgetItem(str(format(int(self.doubleSpinBox_8.value()), ","))))

    self.buylast.setItem(row_count, 11, QTableWidgetItem(str(format(int(self.doubleSpinBox_9.value()), ","))))

    self.buylast.setItem(row_count, 12, QTableWidgetItem(str(format(int(self.doubleSpinBox_10.value()), ","))))

    self.buylast.setItem(row_count, 13, QTableWidgetItem(str(format(int(self.doubleSpinBox_11.value()), ","))))

    self.buylast.setItem(row_count, 14, QTableWidgetItem(str(format(int(self.doubleSpinBox_12.value()), ","))))

    self.buylast.setItem(row_count, 15, QTableWidgetItem(str(format(int(self.doubleSpinBox_13.value()), ","))))

    self.buylast.setItem(row_count, 16, QTableWidgetItem(str(format(int(self.doubleSpinBox_14.value()), ","))))

    self.buylast.setItem(row_count, 17, QTableWidgetItem(str(format(int(self.doubleSpinBox_15.value()), ","))))

    self.buylast.setItem(row_count, 18, QTableWidgetItem(str(format(int(self.doubleSpinBox_16.value()), ","))))

                

    self.doubleSpinBox_1.setValue(0)

    self.doubleSpinBox_2.setValue(0)

    self.doubleSpinBox_3.setValue(0)

    self.doubleSpinBox_4.setValue(0)

    self.doubleSpinBox_5.setValue(0)

    self.doubleSpinBox_6.setValue(0)

    self.doubleSpinBox_7.setValue(0)

    self.doubleSpinBox_8.setValue(0)

    self.doubleSpinBox_9.setValue(0)

    self.doubleSpinBox_10.setValue(0)

    self.doubleSpinBox_11.setValue(0)

    self.doubleSpinBox_12.setValue(0)

    self.doubleSpinBox_13.setValue(0)

    self.doubleSpinBox_14.setValue(0)

    self.doubleSpinBox_15.setValue(0)

    self.doubleSpinBox_16.setValue(0)

```



종목명 or 종목코드를 입력 후 종목 추가를 누르면 해당 종목의 데이터들을 **buylast**에 추가한다.



종목코드, 종목명, 현재가, 매수가격_1, 매수가격_2, ..., 매도수량_4 까지의 파라미터들을 설정하고 마찬가지로 **buylast**에 추가한다.



***


>>> ### searchItem2() - getItemInfo()


거래할 종목을 추가할 때, 서버에 데이터를 전송하는 부분의 메서드이다.



``` python

def getItemInfo(self, new_code) :

    self.k.kiwoom.dynamicCall("SetInputValue(QString, QString)", "종목코드", new_code)

    self.k.kiwoom.dynamicCall("CommRqData(QString, QString, int, QString)", "주식기본정보요청", "opt10001", 0, "100")

```



종목코드를 설정하고 **주식기본정보요청**으로 서버에 데이터를 요청한다.



***


>> ## 선정종목 삭제


**buylast**에 추가한 종목 중 거래를 원하지 않는 종목이 존재할 시 해당 종목을 삭제하는 코드이다.



코드상으로는 다음에 해당한다.



```python

self.Deletecode.clicked.connect(self.deletecode)

```



***


```python

def deletecode(self) :

    x = self.buylast.selectedIndexes()

    self.buylast.removeRow(x[0].row())

```



간단하게 다음과 같이 구현할 수 있다.



UI에 전시만 된 값들이므로 UI에서 제거해주면 된다.


>> ## 데이터베이스 조작


선정한 종목들의 데이터를 데이터베이스에 저장하고, 불러오기 및 삭제를 할 수 있는 코드이다.



미리 저장해놨던 종목들의 데이트를 불러와서 바로 자동매매 가능하도록 즉, 이용에 용이함을 목적으로 구현하였다.



코드상으로는 다음에 해당한다.



```python

self.Getanal_code = []

self.Load_Stock.clicked.connect(self.Load_code)

self.Save_Stock.clicked.connect(self.Save_selected_code)

self.Del_Stock.clicked.connect(self.delete_code)

```



**Getanal_code** 데이터베이스에 저장할 종목들을 임시로 저장하는 리스트이다.



***


>>> ### 데이터베이스 - Load_code()


```python

def Load_code(self) :



    self.Load_Stock.setText("클릭")

    self.Load_Stock.setStyleSheet("Color : red")

    

    if os.path.exists("dist/Selected_code.txt") :

        f = open("dist/Selected_code.txt", "r", encoding="utf8")

        lines = f.readlines()

        

        if not lines :

            msg = QMessageBox()

            msg.setIcon(QMessageBox.Warning)

            msg.setText("파일이 비어있습니다.")

            msg.setWindowTitle("경고")

            msg.exec_()

        

        for line in lines :

            if line != "" :

                ls = line.split("\t")

                code_n = ls[0]

                name = ls[1]

                price = ls[2]

                a1 = ls[3]

                a2 = ls[4]

                a3 = ls[5]

                a4 = ls[6]

                a5 = ls[7]

                a6 = ls[8]

                a7 = ls[9]

                a8 = ls[10]

                a9 = ls[11]

                a10 = ls[12]

                a11 = ls[13]

                a12 = ls[14]

                a13 = ls[15]

                a14 = ls[16]

                a15 = ls[17]

                a16 = ls[18].split("\n")[0]

                

                self.Getanal_code.append([code_n, name, price, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16])

                

        f.close()

```



이전에 데이터베이스를 구현했던 것과 동일한 방식으로 구현한다.



추가되는 항목으로는 감시박스에 추가된 데이터들로 매수가격_1, 매수수량_1, ... 등에 해당한다.



***


>>> ### 데이터베이스 - Save_selected_code()


```python

def Save_selected_code(self) :

    self.Save_Stock.setText("클릭")

    self.Save_Stock.setStyleSheet("Color : red")

    

    for row in range(self.buylast.rowCount()) :

        code_n = self.buylast.item(row, 0).text()

        name = self.buylast.item(row, 1).text()

        price = self.buylast.item(row, 2).text()

        a1 = self.buylast.item(row, 3).text()

        a2 = self.buylast.item(row, 4).text()

        a3 = self.buylast.item(row, 5).text()

        a4 = self.buylast.item(row, 6).text()

        a5 = self.buylast.item(row, 7).text()

        a6 = self.buylast.item(row, 8).text()

        a7 = self.buylast.item(row, 9).text()

        a8 = self.buylast.item(row, 10).text()

        a9 = self.buylast.item(row, 11).text()

        a10 = self.buylast.item(row, 12).text()

        a11 = self.buylast.item(row, 13).text()

        a12 = self.buylast.item(row, 14).text()

        a13 = self.buylast.item(row, 15).text()

        a14 = self.buylast.item(row, 16).text()

        a15 = self.buylast.item(row, 17).text()

        a16 = self.buylast.item(row, 18).text()

        

        f = open("dist/Selected_code.txt", "a", encoding="utf8")

        f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (code_n, name, price, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16))

        f.close()

```



데이터베이스에 선정한 종목들을 추가하고 저장하는 코드이다.



해당 버튼을 클릭하면 "클릭" 으로 바뀌며 데이터베이스에 저장된다.



***


>>> ### 데이터베이스 - delete_code()


```python

def delete_code(self) :

    self.Del_Stock.setText("클릭")

    self.Del_Stock.setStyleSheet("Color : red")

    

    if os.path.exists("dist/Selected_code.txt") : 

        os.remove("dist/Selected_code.txt")

```



데이터베이스에 저장되어 있는 종목들을 삭제한다.



해당 버튼을 클릭하면 "클릭" 으로 바뀌며 데이터베이스의 모든 종목을에 대한 정보들이 삭제된다.



***


>> ## 자동매매 시작


```python

def start_real_auto(self) :

    print("분할 자동 매매 시작하기")

    self.Start_Auto.setText("자동매매 중")

    self.Start_Auto.setStyleSheet("Color : red")

    h6 = Thread6(self)

    h6.start()

```



해당 메서드를 실행함으로 선정한 종목들을 대상으로 자동매매가 시작된다.



자동매매 알고리즘 구현은 Thread6(Qthread_6.py)에 구현한다.



***


>> ## 자동매매 종료


```python

def stop_auto(self) :

    print("자동 매매 종료하기")

    self.Stop_Everything.setText("자동매매 종료")

    self.Stop_Everything.setStyleSheet("Color : red")



    if hasattr(self, 'h6'):

        self.h6.quit()

        self.h6.wait(5000)

```



스레드를 quit() 메서드로 중단시키면 된다.



***


>> ## 결과 출력


간단하게 현재가를 기준으로 손절 및 익절 가격을 설정하고 자동매매를 다음처럼 할 수 있다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/분할_매매_예시.PNG?raw=true)



***

