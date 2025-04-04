---
layout: single
title:  "[AI 부트캠프] 프로젝트 수행을 위한 이론 Python - 기초"
categories: Bootcamp
tag: [#패스트캠퍼스, #패스트캠퍼스AI부트캠프, #업스테이지패스트캠퍼스, #UpstageAILab, #국비지원, #패스트캠퍼스업스테이지에이아이랩, #패스트캠퍼스업스테이지부트캠프]
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


부트캠프가 시작되고 첫 한 주를 보냈다.



실시간으로 강사님이 알려주시는 강의와 녹화된 영상을 수강하는 방식의 온라인 강의가 존재했다.



파이썬 기초 문법, 라이브러리 등에 대해 강의가 진행되었는데 전체 강의 중 나에게 유용했던 부분을 기록에 남기고자 한다.



***


# 1. 온라인 강의


패스트캠퍼스 사이트에 들어가서 강의를 찾아 수강하는 방식으로 진행되었다.



변수 선언부터 제어문, 반복문, 함수, 라이브러리, 클래스 등 기초부터 강의가 진행되었는데 각각의 단원(?)마다 호흡을 짧게두어 분위기를 환기시킬 수 있다는 점이 좋았다.



기초를 진행하는 과정은 사실 대학시절부터 수도없이 썼던 프로그래밍 언어라 익숙하였고, 복습하는 느낌으로 내가 부족했던 부분이 있었는지를 확인하며 들었다.



> - **Matplotlib, Seaborn**을 이용한 데이터 시각화



해당 부분에서 내가 모르고 있었던 시각화 툴, 방법이 있었고 강의를 통해 알게되었다.



특히 이런식으로 시각화를 진행할 수 있다는 점이 놀라웠다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/산점도.PNG?raw=true)




또한, 강의에서 진행되었던 **plot** 이외에도 실제 사용 시 아래의 두 사이트를 많이 참고해야겠구나 생각했다.



> - [Matplotlib](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)



> - [Seaborn](https://seaborn.pydata.org/)



***


# 2. 실시간 강의


개발자로 시작하여 현재 데이터 분석회사 대표이신 김인섭 강사님의 파이썬 기초 실시간 강의가 진행되었다.



아주 기초적인 문법들을 빠르게 진행되었고, 외부 라이브러리를 활용하는 부분부터 조금 천천히 자세하게 알려주셨다.



그 중 크롤링 부분에 있어 집중력있게 들었는데, 내가 지금까지 사용한 크롤링 라이브러리는 **BeautifulSoup**를 주로 사용하여 진행하였고, **Selenium**의 존재도 알고있었으나 BeautifulSoup만으로도 충분하여 많이 사용하지는 않았었다.



그런데 실제로 사용해보니 달랐다.



이런식으로 실제 사이트에 접속하여 동적으로 크롤링을 수행할 수 있었다.



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Bootcamp/크롤링_날씨.PNG?raw=true)




실제로 이런 방식으로 크롤링이란 무기로 데이터를 수집하여 여러 자동화 프로그램들을 배포, 수익으로 연결하셨던 것을 이야기해주셨을 때 다음과 같은 생각이 들었다.



> 내가 지금까지 해왔던 주어진 문제를 해결하는 코딩이 아닌 직접 문제를 발견 및 정의하여 그에 대한 해결책을 찾는, 즉 창의성을 가지고 아이디어를 생각해내서 구현할 수 있을까?



이런 관점에서 돌아보니 점점 스스로 생각하는 부분이 적어졌던 것 같고, 특히 ChatGPT와 같은 **LLM**모델들의 의존도가 많이 올라갔던 것 같았다.


# 파이썬 기초 강의들을 마치며


수도 없이 써왔던 제어문, 클래스, 함수 등 익숙했던 것이었다.



하지만 이런 기초들을 바탕으로 크롤링, 데이터 분석, 머신러닝, 딥러닝 등으로 발전하는 토대가 되기에 소홀히 할 수 없는 부분이기도 하다.



내가 지금까지 Python으로 코딩을 하면서 돌아보는 느낌으로 어떤 점이 부족한지를 중점적으로 생각하며 들었고 실제로 몇몇 부분에서 헷갈렸던 부분들, 미흡했던 부분들이 존재했었다.



예시로, 데이콘 경진대회를 진행할 때 당시에도 EDA 부분에서 미흡했던 부분이 있었다.



> info(), isnull(), unique() 등과 같이 **Numpy** 및 **Pandas** 라이브러리만 주로 이용하여 결측치의 존재 여부, 범주형 및 수치형 특성들 등 전체적인 값 보다는 특정 값에 집중하여 분석을 진행했고, 이를 기반으로 모델을 훈련시켰고 낮은 성적은 아니었지만 극상위권까지 가기 힘들었었다.



만약 EDA 진행하는데에 있어 시각화 라이브러리를 더 잘 활용했었더라면 결과가 조금이라돠 달라지지 않았을까라는 생각도 든다.



앞으로 진행할 여러 강의들, 경진대회, 프로젝트 이외에도 실무에서 기초로 인해 흔들리는 일이 없도록 지속적으로 복습해야겠다.

