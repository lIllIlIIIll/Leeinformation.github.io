---
layout: single
title:  "[프로그래머스] 홀짝트리"
categories: Coding_Test
tag: [python, coding]
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


# 1. 홀짝 트리, 역홀짝 트리 판별


> 문제의 제한사항

    

    - **nodes**의 최대 길이는 400,000

    - **edges**의 최대 길이는 1,000,000



이 두 조건을 보고 내린 결론은 모든 노드를 루트 노드로 가능한 모든 트리를 판별하려고 하면 불가능하겠구나 생각했다.



이는 한 번의 탐색으로 주어진 트리가 홀짝 트리인지, 역홀짝 트리인지를 판별해야한다는 점을 생각하고 문제의 테스트 케이스를 보았다.



***



![image.png](https://github.com/lIllIlIIIll/Leeinformation.github.io/blob/master/_posts/image/Coding_Test/홀짝트리.PNG?raw=true)




***



처음에는 지금까지의 모든 트리들을 부모 노드(루트 노드), 자식노드로 보았기에 구분하여 보는게 어려웠다.



> ***모든 노드를 부모 노드, 자식 노드 구분없이 바라보면...***



    - 9와 연결된 간선의 개수 : 2개 → 역홀수 노드

    - 11과 연결된 간선의 개수 : 1개 → 홀수 노드

    - 7과 연결된 간선의 개수 : 1개 → 홀수 노드



> ***이제 노드들 중 하나를 루트 노드로 생각하여 트리를 분석해 보면...***



    - 9가 루트 노드일 경우

        - 9 : 역홀수 노드

        - 11 : 역홀수 노드

        - 7 : 역홀수 노드



    - 11이 루트 노드일 경우

        - 11 : 홀수 노드

        - 9 : 홀수 노드

        - 7 : 역홀수 노드



    - 7이 루트 노드일 경우

        - 7 : 홀수 노드

        - 9 : 홀수 노드

        - 11 : 역홀수 노드



즉, 루트 노드가 결정되면 다음과 같은 결론을 얻을 수 있다.



- 루트 노드는 간선의 개수가 그대로이므로 역홀짝 or 홀짝 노드의 변화가 없다.

- 나머지 노드들은 간선의 개수가 루트 노드로 연결되어 -1 되므로 다음과 같이 변한다.

    - 홀수 노드 → 역홀수 노드

    - 짝수 노드 → 역짝수 노드

    - 역홀수 노드 → 홀수 노드

    - 역짝수 노드 → 짝수 노드



이와 같은 결론은 **노드들 중 단 하나만 홀짝 노드이면 홀짝 트리, 단 하나만 역홀짝 노드이면 역홀짝 트리가 된다** 의 결론으로 도달한다.





> 예를 들어, 하나의 노드만 짝수 노드이면, 나머지는 역홀수 or 역짝수 노드이므로 짝수 노드를 루트 노드로 지정하면, 나머지는 홀수 or 짝수 노드로 바뀌게 되므로 이는 홀짝 트리가 된다.



이제 이 과정을 코드로 구현하면 다음과 같다!


# 2. 코드 구현


**DFS** 방식을 이용하여 코드를 구현하는데, **odd_even_list** 역홀수, 역짝수, 홀수, 짝수 노드의 개수를 저장하는 리스트를 생성한다.



문제에서 주어진 **nodes**를 **edges**로 구분하여 트리를 생성하고 순회하며 각각의 노드가 어떤 성질의 노드인지를 판별, 리스트에 저장한다.



리스트에 저장된 값을 비교하여 홀짝 트리인지, 역홀짝 트리인지를 판별한다.



```python
def solution(nodes, edges) :
    graph = {node : [] for node in nodes}
    for a, b in edges :
        graph[a].append(b)
        graph[b].append(a)
        
    visited = set()
    answer = [0,0]
    
    def dfs(node) :
        stack = [node]
        odd_even_list = [0, 0, 0, 0]
        # reversed_odd, reversed_even, odd, even
        
        while stack :
            current = stack.pop()
            if current % 2 != 0 :
                if len(graph[current]) % 2 == 0 :
                    odd_even_list[0] += 1
                else :
                    odd_even_list[2] += 1
            else :
                if len(graph[current]) % 2 == 0 :
                    odd_even_list[3] += 1
                else :
                    odd_even_list[1] += 1
                    
            if current not in visited :
                visited.add(current)
                stack.extend(value for value in graph[current] if value not in visited)
        
        return odd_even_list
    
    for node in list(graph) :
        if node in visited : continue
        tree_list = dfs(node)
        
        if tree_list[0] + tree_list[1] == 1 :
            answer[1] += 1
        if tree_list[2] + tree_list[3] == 1 :
            answer[0] += 1
        
    return answer
```
