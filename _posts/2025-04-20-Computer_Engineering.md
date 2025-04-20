---
layout: single
title:  "[AI 부트캠프] 컴퓨터 공학 개론"
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


***



이번 챕터느 어느 대학을 가던지 컴퓨터공학과라면 배우는 내용들을 5일에 걸쳐 실시간으로 진행되었다.



크게 보면 자료구조, 알고리즘, 컴퓨터구조, 운영체제, 데이터베이스 5과목이다.



사실 해당 내용은 2학년 ~ 3학년에 걸쳐 배웠던 내용이었는데 5일이라는 짧은 시간동안 이해하기에는 비전공자에게는 어려운 내용이 될 것으로 예상하며 들었다.



전공자인 나도 배운지 2년이 되가기에 기억이 가물가물한 부분도 있고, 아예 까먹고 있던 부분도 있었다.



특히 알고리즘부분은 주요 알고리즘 DFS, BFS의 흐름만 기억하고 구현 방법이 잘 기억나지 않았고, 아쉬웠던 부분은 탐욕 알고리즘과 같은 다른 알고리즘도 같이 다뤘으면 좋았을 것 같은데 시간 상 안됬을 것 같기에 아쉬웠다.



***


# 1. 자료구조


***



파이썬에서 다루는 자료구조부터, 어떤 알고리즘의 시간 복잡도, 공간 복잡도 등 코드의 시간적, 공간적(메모리) 효율성과 관련된 측면을 다루는 부분이다.



파이썬은 **Collections** 라이브러리의 ***deque()***를 통해 스택, 큐 등을 구현한다.



코딩 테스트 준비를 하다보면 스택, 큐와 관련된 문제들이 나올 때 마다 항상 사용하는 라이브러리이기 때문에 익숙하기도 했고,



개념적으로도 많이 접했기에 어려운 부분은 없었다.



시간 복잡도, 공간 복잡도도 마찬가지로 코딩 테스트를 준비하면서 문제 제한사항(조건)에 맞는 코드를 작성해야 하기 때문에 어떻게 짜야 효율적으로 코드를 작성할 수 있을까라는 고민을 항상 하므로 어렵지 않았다.



해시 테이블(딕셔너리)의 경우도 리스트로 순회하며 접근하기 보단 해시 값으로 접근하는 것이 훨씬 빠르기에 많이 사용하는 자료구조로 사용하고 있었다.



***



이 다음으로 나왔던 ***Counter, defaultdict, heapq*** 모듈들은 다소 생소했던 모듈들이었다.



**Counter**의 경우 몇 번 사용해 보긴 했지만 주로 사용하지 않는 라이브러리였고(보통 sum()집계함수를 이용), **defaultdict**의 경우 사전을 생성할 때, 생성 후 for문으로 돌며 사전에 값을 넣어줬기 때문에 굳이 초기에 기본값이 들어간 사전이 필요없다고 판단했었기에 사용하지 않았었다.



**heapq**도 거의 사용하지 않던 모듈이었는데, 작업 스케줄링과 같은 우선순위 큐 or 완전 이진 트리와 같은 많이 접할 수 있는 곳에서 사용한다는 사실을 알고 너무 고정적인(?), 내가 알고 있는 모듈, 라이브러리만 사용하려고 하는게 아닌가라는 생각이 들었다.



낯선 모듈들은 코딩 테스트 연습하면서 사용해봄으로써 손에 익을 때까지 반복해야겠다.



***


# 2. 알고리즘


***



기본적인 **sorting(정렬)**에 관련된 메서드인 sort(), sorted()를 알려주셨다.



해당 메서드들은 어느 프로젝트 or Feature Engineering 등 다양한 부분에서 굉장히 많이쓰이는 메서드들이다.



어떤 리스트를 정렬하는데 사용되는데, 매개변수들을 줌으로 다양한 방식 or 어떤 값을 기준으로 정렬할 수 있다.



다음은 **DFS, BFS**의 기초가 되는 **그래프**를 알려주셨는데 사실 그래프는 간선, 노드의 개념만 알면 쉽다.



표현도 마찬가지로 딕셔너리를 통해 연결된 노드들을 표현만 해주면 되므로 쉬운데, 이를 응용하는 DFS, BFS 알고리즘은 난이도가 꽤 있는 편이다.



나는 주로 DFS를 문제해결에 많이 사용하는 편인데, 스택 이용이 익숙해있는지라 BFS보다는 DFS에 익숙한 감이 있고, 되도록이면 DFS를 쓰는 습관이 있는데 둘 다 고루 잘 사용할 수 있어야하기에 연습이 필요할 것 같았다.



**DP(동적 계획법)**도 알게 모르게 사용하고 있었다.



사실 DP는 학부 강의 때 배웠었는데 까먹고 있었고, 이번에 강의를 듣게 되며 되새겼는데 코드를 작성하면서 꽤나 많이 사용하고 있었던 것 같다.



이전에 했던 계산을 반복하지 않음으로써 시간을 줄이는 것, 이전 졸업작품 진행할 때, 단어 사전을 생성하며 많이 썼던 방법이었다.



이러한 점이 복습의 장점이 아닐까 싶다.



내가 놓쳤던 부분이나, 미흡한 부분을 확인하고 수정해나가는 부분이 있다는 것, 한번 봐서는 내가 어느 부분에 자신이 있고, 어느 부분이 부족한지 모르기에 복습의 기회가 주어지는게 내 입장에서는 굉장히 좋은 것 같다.



***


# 3. 데이터베이스(DB)




# 4. 컴퓨터 구조 & 운영체제


사실상 내 생각에는 두 과목은 거의 한 과목으로 봐도 무방하다고 본다.





# 5. 네트워크

