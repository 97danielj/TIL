[toc]

## 정수 인코딩(Integer Encoding)

- 각 단어를 고유한 정수에 맵핑시키는 전처리 작업

- 단어를 빈도수 순으로 정렬한 단어 집합을 만들고, 빈도수가 높은 순서로 낮은 정수를 부여

### 1. Dictonary : 사용자가 직접 빈도수 계산

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



### 2. Counter

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

### 3. NLTK 정수인코딩(FreeDist)

- nltk에서 빈도수 계산 도구 FreeDist()

```python
from nltk import FreqDist
import numpy as np
# np.hstack으로 문장 구분을 제거
단어_모음=FreqDist(np.hstack(pr_data)) #빈도수 계산 도구 =Counter랑 비슷
단어_모음 #각 원소(리스트)에 대해 옆으로 붙이기. 1차원 리스트로 사용
```

### 4. Keras의 정수인코딩

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
#tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
#미리 정수 인덱스에 없는 단어들을 <OOV>로 인덱싱 한다. <OOV>의 인덱스는 1
tokenizer.fit_on_texts(preprocessed_sentences) #정수 인덱스 사전 ㅅ


tokenizer.texts_to_sequences(preprocessed_sentences) #실제 적용
#만약 oov_token을 사용하기로 했다면 케라스 토크나이저는 기본적으로 'OOV'의 인덱스를 1로 합니다.
```



