[toc]

# 2. BOW 인코딩

- 단어의 등장 순서를 고려하지 않고 빈도수 기반의 단어 표현 방법인 Bag of Words

## 1. Bag of Words란?

- 단어들의 순서는 전혀 고려하지 않고, **단어들의 출현빈도에만 집중하는 텍스트 데이터의 수치화 표현 방법이다.**

- BoW를 만드는 과정을 이렇게 두 가지 과정으로 생각해보겠습니다.

  ```scss
  (1) 각 단어에 고유한 정수 인덱스를 부여합니다.  # 단어 집합 생성.
  (2) 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만듭니다.  
  ```

**문서1 : 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.**

문서1에 대해서 BoW를 만들어보겠습니다. 아래의 함수는 입력된 문서에 대해서 단어 집합(vocaburary)을 만들어 각 단어에 정수 인덱스를 할당하고, BoW를 만듭니다.

```python
from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
  # 온점 제거 및 형태소 분석
  document = document.replace('.', '') #정제
  tokenized_document = okt.morphs(document) #형태소 토큰화

  word_to_index = {} #단어 순서는 무시한체 단어들의 고유 인덱스를 가지는 사전
  bow = [] #문서가 bow로 벡터화 된다.

  for word in tokenized_document:  
    if word not in word_to_index.keys():
      word_to_index[word] = len(word_to_index)  
      # BoW에 전부 기본값 1을 넣는다.
      bow.insert(len(word_to_index) - 1, 1)
    else:
      # 재등장하는 단어의 인덱스
      index = word_to_index.get(word)
      # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
      bow[index] = bow[index] + 1

  return word_to_index, bow
```



## 2.  Scikit-Learn 문서 전처리 기능

Scikit-Learn의 `feature_extraction` 서브패키지와 `feature_extraction.text` 서브패키지는 다음과 같은 문서 전처리용 클래스를 제공한다.

### 1. DictVectorizer 

- `DictVectorizer`는 `feature_extraction` 서브패키지에서 제공한다.

- 각 단어의 수를 세어놓은 사전에서 BOW 인코딩 벡터를 만든다.
- 이미 단어의 수를 세어놓은 사전이 존재
- 문서 자체가 단어의 빈도수를 가지고있다.

```python
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False) #sparse=False하면 희소행렬 그대로
D = [{'A': 1, 'B': 2}, {'B': 3, 'C': 1}] #이미 각 단어의 빈도수가 정해진 사전이 존재
X = v.fit_transform(D)
X
'''
array([[1., 2., 0.],
       [0., 3., 1.]])
'''
v.features_names_ #단어 사전
v.transform({'C':4, 'D':3}) #D는 사전에 없으므로 제거된다.
'''
array([[0., 0., 4.]])
'''
```



### 2. CountVectorizer 

- `CountVectorizer`는 `feature_extraction.text` 서브패키지에서 제공한다.
- 사이킷 런에서 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer클래스 지원
- 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW 인코딩 벡터를 만든다.
- **텍스트 여러줄을(문서) 인코딩한다.**
- 문서 집합에서 단어의 수가 기록된 사전을 생성한다.
- 단어의 인덱스 순서는 랜덤
- 인수
  - stop_words ='english', list, None
    - stop words 목록.‘english’이면 영어용 스탑 워드 사용.
  - analyzer :  문자열 {‘word’, ‘char’, ‘char_wb’} 또는 함수
    - 토큰화 단위 지정
  - `tokenizer` : 함수 또는 None (디폴트)
    - 토큰 생성 함수
  - ngram_range(min_n, max_n) : 튜플
    - n-그램 범위

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.'] #하나의 문서
vector = CountVectorizer() #단어의 빈도를 Count하여 vector로 만든다.

# 코퍼스로부터 각 단어의 빈도수를 기록
print('bag of words vector :', vector.fit_transform(corpus).toarray()) 

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print('vocabulary :',vector.vocabulary_)

bag of words vector : [[1 1 2 1 2 1]]
vocabulary : {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
#단어의 인덱스는 영문자 단어순
#벡터에서 해당 단어의 인덱스에 빈도수를 기록한다.
#CountVector는 기본적으로 길이가 1이상인 문자에 대해서만 토큰으로 인식-> 정제

#N그램은 단어장 생성에 사용할 토큰의 크기를 결정한다.
#바이그램 : ngram_range=(2, 2)
#2개의 연속된 단어를 하나의 토큰으로 인ㅅ
vect = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
vect.vocabulary_
```

- 영어의 경우 띄어쓰기만으로 토큰화가 수행되기 때문에 문제가 없지만 한국어에 CountVectorizer를 적용하면, 조사 등의 이유로 제대로 BoW가 만들어지지 않음을 의미합니다. 따로 tookenizer을 지정해야 한다.



## 2-2. 불용어를 제거한 BoW 만들기

불용어는 자연어 처리에서 별로 의미를 갖지 않는 단어들이다. **Bow를 사용한다는 것은 그 문서에서 각 단어가 얼마나 자주 등장했는지를 보겠다는거다.** 각 단어에 대한 빈도루를 수치화 하겠다는 것은 결국 텍스트 내에서 어떤 단어들이 중요한지를 보고싶다는 의미를 함축하고 있다. 그렇다면 BoW를 만들때 **불용어를 제거하는 일은 자연어 처리의 정확도를 높이기 위해서 선택할 수 있는 전처리 기법이다.**

영어를 BoW를 만들기 위해 사용하는 CountVectorizer는 불용어를 지정하면, 불용어는 제외하고 BoW를 만들 수 있도록 불용어 제거 기능을 지원하고 있습니다.

- (1) **CountVectorizer에서 제공하는 자체 불용어 사용**

```python
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print('bag of words vector :',vect.fit_transform(text).toarray())
print('vocabulary :',vect.vocabulary_)
-------------------------------------
bag of words vector : [[1 1 1]]
vocabulary : {'family': 0, 'important': 1, 'thing': 2}
```

- **(2) nltk에서지원하는 불용어 사용**

```python
text = ["Family is not an important thing. It's everything."]
from nltk.corpus import stopwords

stop_words = stopwords.words('english') #nltk에서 불용어 리스트 지원
vect = CountVectorizer(stop_words=stop_words)
print('bag of words vector :',vect.fit_transform(text).toarray())
print('vocabulary :',vect.vocabulary_)

```



