[TOC]

# 텍스트 전처리

## 1. 정형, 비정형, 반정형 데이터란?

1. 정형데이터(Structured data)
   1. 정형 데이터는 데이터베이스의 정해진 규칙(Rule)에 맞는 데이터
   2. 들어간 데이터 중에 수치 만으로 의미 파악이 쉬운 데이터들을 보통 말한다
   3. 그 값의 의미를 파악하기 쉽고, 규칙적인 값으로 데이터가 들어갈 경우 정형데이터라고 인식
   4. Gender, Age의 특성의 값
2. 비정형 데이터(Unstructured data)
   1. 정해진 규칙이 없어서 값의 의미를 쉽게 파악하기 힘든 경우
   2. 흔히 텍스트, 음성, 영상과 같은 데이터가 비정형 데이터 범위에 속해있다.
3. 반정형 데이터(Semi-structured data)
   1. 반정형 데이터의  반은 Semi를 의미한다. 즉 완전한 정형이 아니라 약한 정형 데이터라는 것
   2. 대표적으로 HTML이나 XML과 같은 포맷
   3. 이런 범주는 대체로 데이터베이스는 아니지만 스키마를 가지고 있는 형태이다.
   4. 데이터베이스 데이터 -> dump - > Json, XML 형태 포맷하는 순간 반정형 데이터?
      1. 구조의 차이
         1. 데이터베이스는 데이터를저장하는 장소와 스키마가 분리되어 있다.
         2. 데이터베이스는 테이블을 생성하고, 데이터를 저장
         3. 반정형 은 한 텍스트 파일에 Coulmn과 Values를 모두 출력한다.
         4. 반정형을 정형, 비정형에서 완벽 구분은 어렵다.



## 2. 텍스트 전처리

- 말뭉치(코퍼스)
  - 말뭉치 또는 코퍼스(복수 : corpora)
  - __자연언어 연구를 위해 특정한 목적을 가지고 언어의 표본을 추출한 집합__
  - 코퍼스 분석 뿐만 아니라 언어 분석에 사용되는 실제 언어의 체계적 디지털 모음
  - 둘 이상의 코포스는 코포라
- 텍스트 전처리
  - 자연어 처리에서 크롤링 등으로 얻어낸 코포스 데이터가 전처리되지 않은 상태시 해당 데이터를 용도에 맞게 토큰화 & 정제 & 정규화를 진행 해야 함
- 테이터 분석
  - 데이터 수집(추출)
  - 데이터 전처리
    - 토큰화
    - 정제
    - 정규화



## 3. 토큰화

- 단어 토큰화(Word Tokenization)
- 문장 토큰화(Sentence Tokenization)



## 4. 단어 토큰화(Word Tokenization)

- 토큰 기준을 단어로 하여 토큰화 하는것
- 단어(word)는 단어 단위 외에도 단어구, 의미를 갖는 문자열로도 간주됨
- 보통 토큰화 작업은 단순한 구두점이나 특수문자를 전부 제거한 정제(cleaning)작업을 수행하는 것만으로 해결되지 않음
- 구두점이나 특수문자를 전부 제거하면 어떤 토큰은 의미를 잃어 버리는 경우가 발생함
- 띄어쓰기 단위로 자를시 단어 토큰 구분이 망가지는 언어도 존재
  - ex) 데이터 과학
- 토큰화 진행 시, 예상하지 못한 경우가 있어서 토큰화의 기준을 생각해봐야 하는 경우가 발생함 Ex) 영어의 어퍼스트로피 토큰화 문제 Don't

|           코드 및 모듈           |                             기능                             |
| :------------------------------: | :----------------------------------------------------------: |
|  nltk(Natural Language Toolkit)  |          자연어(영어) 처리 및 문자열 분석용 패키지           |
| konlpy(Korean Natural Language ) |             한국어 자연어 처리 및 분석용  패키지             |
|  kss(Korean Sentence Splitter)   | 한국어 문장 구분기 제공 패키지(**New Line(\n)을 포함한다.**) |

```python
from nltk.tokenize import word_tokenize
# 가장 많이 사용하는 자연어(영어) 단어 토큰화기. 형태소 기준 으로 분류

from nltk.tokenize import WordPinctTokenizer
#또 다른 단어 토큰화기 클래스를 호출. 단어 기준으로 분류

from tensorflow.keras.preprocessing.text import text_to_word_sequence
#keras의 텍스트 전처리 패키지에서 text 단어 분리기. 단어를 모두 소문자 취급

TreebankWordTokenizer().tokenize(text)
#nktl의 또 다른 단어 토큰화기 = word_tokenize()와 동일하다.

from nltk.tag import pos_tag #pos(part of sentence = 품사)
t_t = word_tokenize(t6) #단어 토큰화
pos_tag(t_t) #토큰화된 리스트의 각 원소에 품사 태그 붙인다.

```



- 토큰화 고려 사항 
  - 구두점이나 특수 문자를 단순 제외해서는 안 된다.
    - 구두점조차 하나의 토큰으로 분류하기도 한다.
    - 마침표(.)
      - 문장의 경계를 통한 단어를 추출용 기준 이용
    - 단어 자체 특수문자나 구두점
      - m.p.h Ph.D나 AT&T 특수 문자의 달러나($)나 슬래시(/)
    - 숫자 사이에 컴마(,)
      - 123,456,789
  - 줄임말과 단어 내에 띄어쓰기가 있는경우 
    - what're 는 what are의 줄임말
    - we're는 we are의 줄임말
    - New york은 단어 자체에 띄어쓰기가 존재
- Penn Treebank Tokenization의 규칙
  - 영어 토큰화 표준으로 사용되는 규칙(=word_tokenization)
  - 규칙1 : 하이푼으로 구성된 단어는 하나로 유지
  - 규칙2 : doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리



## 3. 문장 토큰화(Sentence Tokenization)

- 문장 단위로 구분하는 작업으로 때로는 문장 분류

- 정제되지 않은 상태라면, 코퍼스는 문장 단위로 구분되어 있지 않아서 이를 사용하고 하는 용도에 맞게 문장 토큰화가 필요.

- !나 ?는 문장의 구분을 위한 꽤 명확한 구분자(boundary) 역할

- 마침표. 는 문장의 끝이 아니더라도 등장할 수 있음=> 단어 자체의 . 이 들어갈 수 있다.

  - EX1) IP 192.168.56.31 서버에 들어가서 aaa@gmail.com로 굙허 좀 보냐줘
  - EX2) Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.

- ```python
  #문장 토큰화기(영어)
  from nltk.tokenize import sent_tokenize
  sent_tokenize(t3)
  
  #문장 토큰화기(한글)
  import kss #Korean Sentence Spliter
  kss.split_sentences(text
  ```



## 4. 한국어 토큰화

- 영어는 New York과 같은 합성어나 he's와 같이 줄임말에 대한 예외처리만 한다면, 띄어쓰기 기준으로 띄어쓰기 토큰화를 수행해도 단어토큰화가 잘 작동
- 한국어는 영어와 달리띄어 쓰기만으로는 토큰화를 하기에 부족
- 한귝어의 경우 띄어쓰기 단위가 되는 단위를 어절이라고 하는대 어절 토큰화는 한국어 NLP에서 지양되고 있음
- 어절토큰화와 단어토큰화는 다름
- 한국어가 영어랑 다른 형태를 가지는 언어인 교착어라는 점에서기인
- 교착어란 조사, 어미 등을 붙여서 말을 만드는 언어를 말함
  - 조사가 존재
    - 그라는 단어 하나에도 '그가', '그에게', '그를' ,'그와', '그는'과 같이 다양한 조사가 '그'라는 글자 뒤에 띄어쓰기 없이 바로 붙게됨
    - 대부분의 한국어 NLP에서 조사는 분리해줄 필요가 있음
  - 띄어쓰기 단이가 영어처럼 독립적인 단어가 아님
  - 한국어는 어절이 독립적인 단어로 구성되는 것이 아니라 조사 등의 무언가가 붙어있는 경우
    가 많아서 __이를 전부 분리해줘야 함__

## 5. 형태소

- 한국어 토큰화에서는 형태소란 개념을 반드시 이해해야 함
- 형태소
  - 뜻을 가진 가장 작은 말의 단위
  - 이 형태소에는 두가지 형태소가 있는데 자립과 의존
  - 한국어의 문장=> 어절(띄어쓰기 단위)로 구성 =>  어절은 다양한 형태소로 구상
- 자립 형태소
  - 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소
  - __그 자체로 단어가 됨__
  - 체언, 수식언, 감탄사
- 의존 형태소
  - 다른 형태소와 결합하여 사용되는 형태소
  - 접사, 조사, 어미, 어간
- 한국어에서 영어에서의 단어 토큰화와 유사한 형태를 얻어내려면 어절토큰화가 아난 형태소 토큰화를 수행해야 한다.

```python
from konlpy.tag import Okt #세종 품사 사전. 오픈 코리언 태그 형태소
from konlpy.tag import Kkma #서울대 제작 사전

nl = Okt() #형태소 토큰화기 객체
n1.morphs(text) #텍스트에서 형태소 반환한다.
n1.pos(text) #텍스트에서 품사 정보를 부착하여 반환한다.
#영어 단어 토큰화에서는 pos_tag(토큰화된 리스트)

n1.nouns(text)

n2=Kkma() #어절,조사,접사 모두 형태소로 분리
n2.morphs(t7),n2.pos(t7),n2.nouns(t7)

```

- 품사 태깅(part-of-speech tagging) – 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해놓는 작업



## 6. 정제(cleaning)

- 갖고 있는 코퍼스로부터 노이즈 데이터를 제거
- 완벽한 정제작업은 어렵다.
- 일종의 합의점을 찾음



## 7. 정규화(nomalization)

- **표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들기**
- **규칙에 기반한 통합**
  - 정규화 규칙의 예로서 같은 의미를 갖고있음에도, 표기가 다른 단어들을 하나의 단어로 정규화하는 방법을 사용
- **대,소문자 통합**
  - 대부분의 글은 소문자로 작성되기 때문에 대.소문자 통합 작업은 대부분 대문자를 소문자로 변환하는 소문자 변환작업
- **불필요한 단어의 제거**
  - 등장빈도가 적은 단어
  - 길이가 짧은 단어
- 단어의 개수 줄일 수 있는 기법
  - 표제어 추출
  - 어간 추출
- **하나의 단어로 일반화시킬  수 있다면 하나의 단어로 일반화 시켜서 문서 내의 단어 수를 줄이겠다는 것**
- __단어의 빈도수를 기반으로 문제를 풀고자 하는 자연어 처리문제에 주로 사용__
- 자연어 처리에서 전처리 정규화의 지향점은 언제나 복잡성을 줄이는 일
- 자연어 처리 => 토큰화 => 정규화(통합) -> 빈도수기반 자연어 처리 문제 해결

### 1. 표제어 추출

- '표제어' 또는 '기본 사전형 단어' 정도의 의
- 표제어 추출은 단어들이 다른 형태를 가지더라도, 그 뿌리 단어를 찾아가서 단어의 개수를 줄일 수 있는지 판단
- 표제어 추출을 하는 가장 섬세한 방법은 단어의 형태학적 파싱을 먼저 진행
- 어간 : 단어의 의미를 담고 있는 단어의 핵심 부분
- 접사 : 단어에 추가적인 의미를 주는 부분
- 형태학적 파싱은 이 두가지 구성요소를 분리하는 작업



#### 1. 어간 추출

- 어간(Stem)을 추출하는 작업
- 형태학적 분석을 단순화한 버전
- 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업
- 섬세한 작업이 아니기 때문에 어간 추출 후에 나오는 결과 단어는 사전에 존재하지 않는 단어 일수도 있다.
- 한국어는 5언 9품사의 구조를 가지고 있다.
  - 이중 동사와 형용사는 어간(stem)고 어미(ending)의 결합으로 구성
- 동사변화
  - 용언의 어간이 어미를 갖는 일
  - 규칙 불규칙 형이 있음
- 어간
  - 용언을 활용할 때, 원칙적으로 모양이 변하지 않는 부분. 활용에서 어미에 선행하는 부분. 때로는 어간의 모양도 바뀔수 있다.
- 어미
  - 용언의 어간 뒤에 붙어서 활용하면서 변하는 부분, 여러 문법적 기능 수행
- 규칙 동사변화
  - 어간이 어미를 취할 때, 어간의 모습이 일정
  - 단순히 분리해 주면 어간이 추출이 됨
- 불규칙 동사변화
  - 어간이 어미를 취할 때 어간의 모습이 바뀌거나 취하는 어미가 특수한 어미일 경우
  - 단순히 분리만으로 어간 추출이 어렵고 복잡한 규칙을 필요

```python
import nltk
nltk.download('wordnet') #단어그물(가운데 기본 사전형 단어)
nltk.download('stopwords') #불용어 사전 다운
nltk.download('punkt') #문장토큰화기 다운

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

from nltk.stem import WordNetLemmatizer #표제어 모델
f=WordNetLemmatizer().lemmatize #표제어 추출
#표제어 추출(근원). 단어의 형태가 적절히 보존되는 양상.
[f(x) for x in words] #하나씩 단어 하나씩 표제어 추출
#표재어 추출기가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문입니다.
lemmatizer.lemmatize('dies', 'v')
lemmatizer.lemmatize('watched', 'v')

#어간 추출은 섬세한 작업이 아니기 때문에 어간 추출 후에 나오는 결과가 사전에 존재하지 않을수도 있다.
from nltk.stem import PorterStemmer #포토 어간 추출 알고리즘

```

- 어간 추출 속도는 표제어 추출보다 일반적으로 빠른데, 포터 어간 추출기는 정밀하게 설계되어 정확되가 높으므로 영어 자연어 처리에서 어간 추출시 가장 많이 선택

```python
from nlkt.stem import PorterStemmer
#포터 알고리즘
#대문자-> 소문자(대소문자 통합)
#ALIZE → AL
#ANCE → 제거
#ICAL → IC
from nltk.stem import LancasterStemmer
#랑커스터 스테머 알고리즘
#대문자-> 소문자(대소문자 통합)
```

- 코퍼스에 스태머를 적용시 어떤 스태머가 해당 코퍼스에 적합한지를 판단한 후에 사용
- 지나친 일반화
  - organization->organ은 완전히 다른 단어 . 하지만 organ과 같이 정규화가 된다.



## 8. 불용어

- __실제 의미 분석을 하는데 거의 기여하는 바가 없는 단어들__
- I, my, me, over, 조사, 접미사는 자주 문장에서 등장하나 실제 의미 분석을 하는데 기영 X
- 한국어에서 불용어를 제거하는 방법으로 간단하게는 토큰화 후에 __조사, 접속사__ 등을 제거
- 사용자가 직접 불용어 사전을 만들게 되는 경우가 많다.
- 불용어가 많은 경우에는 코드 내에서 직접 정의하지 않고 txt파일이나 csv 파일로 정리해놓고 이를 불러와소 사용하기도 함. 즉, 따로 불용어 사전을 직접 제작이 아닌 txt나 csv파일로 불러온다.

```python
from nltk.corpus import stopwords #불용어 패키지 인풋
stop_words_list = stopwords.words('english') #영어 불용어 사전 호출
len(stop_words_list)
```





## 9. 정규표현식

- 스크레핑에서 학습한 정규표현식을 확인

| 메타문자 |                             의미                             |
| :------: | :----------------------------------------------------------: |
|    .     |         한개의 임의의 문자를 나타냅니다.('\n 제외')          |
|    ?     |  앞의 문자가 존재 할수도 있거, 존재하진 않을 수도 있습니다.  |
|    *     |             앞의 문자가 무한개로 존재 또는 존재X             |
|    []    | 대괄호 안의 문자들 중 한개의 문자와 매치<br />[a-z]와 같이 범위를 지정할 수도 있습니다. |
| [^문자]  |            해당 문자를 제외한 문자를 매치합니다.             |
|   AIB    |                  A또는 B의 의미를 가집니다                   |
|    \d    |                      숫자와 매치. [0-9]                      |
|    \D    |                    숫자가 아닌 것과 매치                     |
|    \s    |                      whitespace와 매치                       |
|    \w    |                       문자+숫자와 매치                       |
|    \W    |                     특수문자+공백과 매치                     |



- 정규표현식 모듈 함수

|     함수      |                             기능                             |
| :-----------: | :----------------------------------------------------------: |
| re.compile()  |     정규표현식을 컴파일하는 함수입니다.(패턴으로 컴파일)     |
|  re.search()  |  문자열 전체에 대해서 정규표현식과 매치되는지를 검색합니다   |
|  re.match()   |     문자열의 처음이 정규표현식과 매치되는지를 검색합니다     |
|  re.split()   | 정규 표현식을 기준으로 문자열을 분리하여 리스트로 리턴합니다. |
| re.findall()  | 문자열에서 정규 표현식과 매치되는 모든 경우의 문자열을 찾아서 리스트로 리턴합니다. 만약, 매치되는 문자열이 없다면 빈 리스트가 리턴됩니다. |
| re.finditer() | 문자열에서 정규 표현식과 매치되는 모든 경우의 문자열에 대한 이터레이터 객체를 리턴합니다. 이터레이블 객체를 반환 |
|   re.sub()    | 문자열에서 정규 표현식과 일치하는 부분에 대해서 다른 문자열로 대체합니다 |

```python
from nltk.tokenize import RegexpTokenizer
ck1=RegexpTokenizer('[\w]+') #정규표현식 토튼화기
ck1.tokenize(t)
```



## 10. 정수 인코딩(Integer Encoding)

- 각 단어를 고유한 정수에 맵핑시키는 전처리 작업

- 단어를 빈도수 순으로 정렬한 단어 집합을 만들고, 빈도수가 높은 순서로 낮은 정수를 부여

- ### 1. Dictonary : 사용자가 직접 빈도수 계산

```python
단어_모음={} #정수 인코딩(맵핑)하기 위한 빈도수 딕셔너리
pr_data=[]
불용성단어 = set(stopwords.words('english'))
for 문장 in 문장_토큰화_리스트: #한 문장식
    단어_토큰화_리스트 = word_tokenize(문장)
    l=[]
    for 단어 in 단어_토큰화_리스트:
        소문자화_된_단어 = 단어.lower() # 정규화2
        if 소문자화_된_단어 not in 불용성단어:
            if len(소문자화_된_단어) > 2: #단어수 2이하 제거. 노이즈제거
                l.append(소문자화_된_단어)# 전처리된 단어
                if 소문자화_된_단어 not in 단어_모음:
                    단어_모음[소문자화_된_단어] = 0 #빈도수 딕셔너리에 단어 초기화
                단어_모음[소문자화_된_단어]+=1
    #한 문장을 다 돌았다면 문장 전처리 단어 l을 pr_data에 추가
    pr_data.append(l)
pr_data
```



- ### 2. Counter

  - Counter 모듈을 사용.
  - 중복은 제거하고 단어의 빈도수를 기록한다.
  - Counter 객체로 반환

```python
단어_모음집 = sum(pr_data,[]) #일차원 리스트
단어_모음집

결과_단어_모음집=Counter(단어_모음집) #중복은제고 단어의 빈도수를 기록
결과_단어_모음집

top=4 #빈도수중 상위 5개 
빈도수별_단어=결과_단어_모음집.most_common(top)
빈도수별_단어

단어_인덱스2={} #빈도수가 높을 수록 낮은 정수 인덱스 부여
i=0
for 단어,빈도수 in 빈도수별_단어:
    i+=1
    단어_인덱스2[단어]=i
단어_인덱스2

#Tip) 텍스트 매핑시 존재 하지 않는 단어에 대해서 OOV에러가 날수도 있으니 정수_인덱스['OOV'] = len(정수_인덱스)+1
```

- ### 3. NLTK 정수인코딩(FreeDist)
  
  - nltk에서 빈도수 계산 도구 FreeDist()

```python
from nltk import FreqDist
import numpy as np
# np.hstack으로 문장 구분을 제거
단어_모음=FreqDist(np.hstack(pr_data)) #빈도수 계산 도구 =Counter랑 비슷
단어_모음 #각 원소(리스트)에 대해 옆으로 붙이기. 1차원 리스트로 사용
```

- ### 4. Keras의 텍스트 전처리
  
  - 케라스는 기본적인 전처리를 위한 도구들을 제공한다.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
preprocessed_sentences #앞서 문장과 단어 토큰화된 데이터
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences) 
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
#빈도수 높은 순으로 낮은 정수 인덱스를 부여
print(ck_t.word_index) #생성된 정수_인덱스

ck_t.word_counts #OrederedDict()
print(tokenizer.texts_to_sequences(preprocessed_sentences))
#입력으로 들어온 코퍼스에 대해서 각 단어를 이미 정해진 인덱스로 변환합니다.

vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1)
# 상위 5개 단어만 사용
#num_words에 +1하는 이유는 num_words는 숫자를 0부터 카운트 합니다.
#정수 인덱스 0이 존재하지 않는데 단어 집합의 산정 이유는 패딩이라는 작업떄문에
tokenizer.fit_on_texts(preprocessed_sentences) #컴파일
tokenizer.texts_to_sequences(preprocessed_sentences) #실제 적용
#만약 oov_token을 사용하기로 했다면 케라스 토크나이저는 기본적으로 'OOV'의 인덱스를 1로 합니다.
```



## 11. 패딩(Padding)

- 병렬 연산을 위해서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업
- 하나의 행렬로 만듬
- 기준 길이는 토큰화된 행렬에서 가장 긴 문장의 길이로 맞춘다.
- 채워주는 단어(가상의 단어)는 'PAD'라는 0번 정수 인덱스를 가지는 가상 단어를 사용

### 1. numpy 패딩

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
#인코딩

#최대길이
max_len = max(len(item) for item in encoded)

#제로패딩 : 0을 채워서 데이터의 크기를 조정
```

### 2. 케라스 전처리 도구로 패딩하기

- 기본적으로 앞에 0을 채운다.
- 앞에 채우고 싶으면 padding = 'post'

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
#케라스는 패딩을 위한 pad_sequence()를 제공하고 있습니다.

tk = Tokenizer()
tk.fit_on_texts(pr_data) #정수 인덱스 사전
encoded = tk.texts_to_sequences(pr_data) #인코딩
encoded

end_data=pad_sequences(encoded,padding='post') #encoded리스트를 행렬로 패딩. 기존데이터는 전방에
end_data

#뒤에서 단어 삭제
end_data2=pad_sequences(encoded,padding='post',truncating='post', maxlen=5)
#padding값을 바꿀시 존재하는 정수 인덱스와 겹치치 않게 정수를 부여한 후 value=v로 함수에 전달.
```

