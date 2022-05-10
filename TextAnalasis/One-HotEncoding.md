[TOC]

# 원-핫 인코딩

## 1. 개념

- 컴퓨터는 자연어 처리에서 문자를 숫자로 바꾸고, 그 기법은 여러가지가 있다.
- 많은 기법 중 원-핫 인코딩은 그 많은 기법 중에서 단어를 표현하는 가장 기본적인 표현방법.
- 단어집합 : 서로 다른 단어들의 집합
  - 기본적으로 book과 books와 같이 단어의 변형 형태도 다른 단어로 간주
- 원-핫 인코딩 : 문자를 숫자로 처리. __문자를 벡터로 바꾸는 것__



## 2. 순서

1. 단어 집합을 만들기(중복은 제거)
2. 단어 집합에 고유한 정수를 부여하는 정수 인코딩

---------------정수 인코딩------------

1. 단어집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 고유한 정수를  인덱스로 간주하고 해당 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여합니다.

```python
#1. numpy로 직접 원-핫 인코딩
#해당 단어의 정수 인덱스를 추출 후 원-핫 인코딩 한다.(벡터 반환)
def one_hot_encoding(word, word_to_index): #단어 정수 인덱스
  one_hot_vector = [0]*(len(word_to_index)) #차원이 6개인 벡터
  index = word_to_index[word] #해당 정수 인덱스를 추출
  one_hot_vector[index] = 1 #벡터에 해당 정수 인덱스를 1로
  return one_hot_vector

#2. 케라스를 이용한 원-핫 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text]) #벡터에서 정수인덱스 생성
print('단어 집합 :',tokenizer.word_index)

#정수 인코딩된 결과로 부터 원-핫인코딩을 수행. 단어의 개수만큼 벡터의 개수 반환
encoded = [2, 5, 1, 6, 3, 7]
one_hot = to_categorical(encoded)
print(one_hot)

```



## 3. 원-핫 인코딩의 한계

- 단어의 개수가 늘어날 수록, 벡터의 차원이 늘어나 필요 공간이 계속 늘어난다.
  - ex) 1000개의 단어 코퍼스=> 모든 단어는 1000개의 차원을 가지는 벡터
- 단어의 유사도를 표현하지 못한다는 단점
  - 검색시스템에서 문제가 될수 있다.
  - 연관 검색어를 보여줄수 없다. 단어간 유사성을 계산 할 수가 없어서
- 해결방안
  - 단어의 잠재 의미를 반영하여 다차원 공간에 벡터화 하는 기법
  - 카운트 기반의 벡터화 방법 : LSA,  HAL
  - 예측 기반으로 벡터화 방벙 : NNLM, RNNLM, Word2Vec, FastText