---
layout: single
title:  "[키움 자동매매 프로그램]"
categories: Kiwoom
tag: [python, coding, API]
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


키움 OpenAPI를 사용하기 위해선 우리가 현재 사용하고 있는 64bit 환경이 아닌 32bit 환경이 필요하다.



anaconda와 키움 OpenAPI가 설치되었다면 32bit의 가상환경을 구성한다.



물론 자신의 컴퓨터가 원래 32bit 환경이라면 밑의 설정을 따로 할 필요는 없다.


- Anaconda Prompt를 열어 다음의 명령어를 차례로 입력하면 32bit의 개발환경이 구축된다.



    - conda config --env --set subdir win-32

    - conda create -n pykiwoom_32 python=3.8

    - conda activate pykiwoom_32

    - conda config --env --set subdir win-32

    - conda install python=3.8

    - python

    - import platform

    - print(platform.architecture())

    - exit()

    - conda deactivate

    - conda config --env --set subdir win-64


***


각각의 명령어가 어떤 의미인지 알아보면



1. conda config --env --set subdir win-32



(base) 환경의 bit를 32bit 환경으로 전환



전환 후 **conda info** 명령어를 통해 32비트 환경으로 바뀌었는지 확인할 수 있다.


![image.png](/image/conda info.png)



**platform : win-32**로 나오면 성공



2. conda create -n pykiwoom_32 python=3.8



가상환경을 생성하는 명령어로, pykiwoom_32 부분은 자신이 원하는 가상환경 이름으로 바꿔도 무방하다.



python 버전은 3.8로 사용한다. (높은 버전을 쓰면 키움 OpenAPI 몇몇의 기능이 작동하지 않는 현상이 발생)



3. conda activate pykiwoom_32



위에서 만들었던 pykiwoom_32 가상환경으로 들어간다.



해당 명령어를 실행하면 (base) → (pykiwoom_32) 로 바뀐 것을 볼 수 있다.


![image.png](/image/conda actifvate pykiwoom_32.png)



4. conda config --env --set subdir win-32



외부환경(base)에서 32bit로 설정하였으므로 내부환경(pykiwoom_32)도 마찬가지로 32bit 환경으로 설정해준다.



5. conda install python=3.8



파이썬을 3.8 버전으로 다시 설치해준다.



6. python(python 실행 부분)



python 명령어로 파이썬을 실행하고,



import platform



print(platform.architecture())



위의 코드로 32bit의 환경인지 확인한다. 다음과 같이 나오면 성공


![image.png](/image/python 명령어.png)



7. exit()



python 코드를 실행할 수 있는 콘솔 환경에서 나간다.



8. conda deactivate



현재의 가상환경(pykiwoom_32)에서 기본 환경(base)로 돌아간다.



9. conda config --env --set subdir win-64



원래의 환경이 64bit 환경이므로 32bit 환경에서 64bit 환경으로 바꿔준다.

