[toc]

# BOW 인코딩

## 1. 개념

- 문서를 숫자 벡터로 변환하는 가장 기본적인 방법은 BOW(Bag of Word)인코딩 방법이다.
- BOW인코딩 방법에서 전체 문서{d1,d2,…,dn}를 구성하는 고정된 단어장 {t1,t2,…,tm}를 만들고 di라는 개별 문서에 단어장에 해당하는 단어들이 포함되어 있는지를 표시하는 방법이다.
- **xi,j=문서 di내의 단어 tj의 출현 빈도**



## 2.  Scikit-Learn 문서 전처리 기능

Scikit-Learn의 `feature_extraction` 서브패키지와 `feature_extraction.text` 서브패키지는 다음과 같은 문서 전처리용 클래스를 제공한다.

- DictVectorizer : 
  - 각 단어의 수를 세어놓은 사전에서 BOW 인코딩 벡터를 만든다.
  - 이미 단어의 수를 세어놓은 사전이 존재
- CountVectorizer :
  - 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW 인코딩 벡터를 만든다.
  - 문서 집합에서 단어의 수가 기록된 사전을 생성한다.
- TfidVectorizer : 
  - CountVectorer와 비슷하지만 TF-IDF 방식으로 단어의 가중치를 조정한 BOW 인코딩 벡터를 만든다.
- [`HashingVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html):
  - 해시 함수(hash function)을 사용하여 적은 메모리와 빠른 속도로 BOW 인코딩 벡터를 만든다.

### 1. DictVectorizer

- `DictVectorizer`는 `feature_extraction` 서브패키지에서 제공한다. 문서에서 단어의 사용 빈도를 나타내는 딕셔너리 정보를 입력받아 BOW 인코딩한 수치 벡터로 변환한다.
- **출현빈도를 딕셔너리 정보로 입력해야한다.**



```python
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D=[{'A':1,'B':2},{'B':3,'C':1}]
#중괄호는 하나의 문장, 하나의 문장의 키값은 단어
#3종 단어롤 이루어진다. 벡터의 차원의 3. 각 단어는 인덱스를 나타내고 벡터의 xij은 그 단어의 빈도수
#각 단어의 빈도수를 미리 알아야 사용가능
X = v.fit_transform(D) #D를 행렬로 변환
print(X)
print(type(X))

#벡터구성 차원
v.feature_names_ 
```



### 2. CountVectorizer

1. **문서를 토큰 리스트로 변환한다.** - fit()
2. 각 문서에서 토큰의 출현 빈도를 센다. fit_transforn()
3. 각 문서를 BOW 인코딩 벡터로 변환한다.

- CounterVectorizer => BOW 인코딩 벡터 문서의 집합-> 단어토큰생성-> 각단어의 수

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',
]

v1=CountVectorizer()
#문장 토큰화가 완료되어야 한다.
v1.fit(corpus)
#학습하면 토큰화-> 출현빈도를 센다.
v1.vocabulary_

v1.transform(['This is the first document. This is the man']).toarray()
#out : array([[0, 1, 1, 2, 0, 0, 0, 2, 0, 2]], dtype=int64)
```

