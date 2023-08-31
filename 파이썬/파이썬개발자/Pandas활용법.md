[toc]



# Pandas 활용법

---

## 시작

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 1. 데이터 오브젝트 생성

- 데이터 오브젝트: 데이터를 담고 있는 그릇
- Series: 1차원 배열로 데이터를 담고 있다. - 인덱스에 의해 저장
- DataFrame: 2차원 배열로 데이터를 담고있다. - 인덱스와 컬럼 기준에 의해 데이터 저장

### Series 객체의 생성

```python
s = pd.Series(['a', 'b', 'c', 'd'])
```

### DataFrame 객체 생성

```python
# numpy 2차원 배열로 생성
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A','B','C','D'])
# 딕셔너리로 생성
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'})
```

DataFrame의 컬럼들은 각기 특별한 자료형을 갖고 있을 수 있습니다. 이는 DataFrame내에 있는 dtypes라는 속성을 통해 확인 가능합니다. 소수점은 float64 로 잡히고, 기본적은 문자열은 str 이 아니라 object 라는 자료형으로 나타납니다.

## 2. 데이터 확인하기

- head() : 첫 5개의 행

- tail() : 마지막 5개의 행을 보여준다.

- ```python
  df.index
  df.columns
  df.values
  df.decribe() # 생성된 DataFrame의 간단한 통계정보
  df.T # 열과 행을 바꾼 형태의 데이터프레임입니다.
  ```

## 3. 데이터 선택하기

데이터프레임 자체가 가지고있는 []슬라이싱 기능을 이용합니다.

데이터프레임 자체가 갖고 있는 슬라이싱은 `df[컬럼명]`, `df[시작인덱스:끝인덱스+1]`, `df[시작인덱스명:끝인덱스명]` 의 형태로 사용할 수 있습니다.

```python
# 열 가져오기
df['A']
# 행가져오시 -> 행범위를 작성
df[1:3]
df['20230827':'20230828']
```

### 이름을 이용하여 선택하기: .loc

라벨의 이름을 이용하여 선택할 수 있는 .loc를 이용할 수도 있습니다.

```python
# 첫 번째 인덱스 값에 해당하는 모든 컬럼의 값 가져오기
df.loc['20130101']
# 컬럼 ‘A’와 컬럼 ‘B’에 대한 모든 값 가져오기.
df.loc[:['A','B']]'
# 행과 컬럼 조건에 맞는 데이터 선택하기
df.loc[dates[0]:dates[1],['A','B']]
```

### 위치를 이용한 선택하기 : .iloc

주의점 : 숫자 슬라이싱이니 마지막 인덱스는 안가져 온다.

```python
df.iloc[3]
df.iloc[1:3,:2]
```

### 조건을 이용하여 선택하기

```python
df[df.A > 0]
# 필터링을 사용하는 경우
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']

# isin([])으로 행 선택
df2[df2['E'].isin(['two', 'three'])]
```

### 데이터 변경하기

우리가 선택했던 데이터 프레임의 특정 값들을 다른 값으로 변경할 수 있습니다. 이에 대한 방법을 알아봅니다.

기존 데이터 프레임에 새로운 열을 추가하고 싶을 때는 다음과 같이 같은 인덱스를 가진 시리즈 하나를 데이터 프레임의 열 하나를 지정하여 넣어 줍니다.

```python
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
df['F'] = s1

#데이터 프레임의 특정 값 선택하여 변경
df.at[dates[0], 'A'] = 0
```



## 4. 결측치 (Missing Data)

pandas에서는 결측치를 np.nan으로 나타냅니다. pandas에서는 결측치를 기본적으로 연산에서 제외

재인덱싱(reindex)는 해당 축에 대하여 인덱스를 변경/추가/삭제를 하게 됩니다. 이는 복사된 데이터 프레임을 반환합니다.

```python
df1 = df.reinex(index=dates[0:4], columns=list(df.comlumns)+['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1

# 결측값 제거하기
df1.dropna(how='any')
# 다른 값으로 채우기
df1.fillna(value=5)
# 결측치 확인
pd.isna(df1)
```

## 5. 연산 (Operations)

### 통계적 지표들 (Stats)

평균 구하기. 일반적으로 결측치는 제외하고 연산을 합니다.

```python
# 평균 구하기, 각 칼럼들에 대해서 구해진다.
df.mean()

# 다른 축에 대해서 평균 구하기 인덱스 기준
df.mean(1) 

# 서로 차원이 달라 인덱스를 맞추어야 하는 두 오브젝트 간의 연산
# 맞추는 축만 지정하면됨
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
df.sub(s, axis='index')
```



### 함수 적용하기 (Apply)

데이터프레임에 함수를 적용할 수 있습니다. 기존에 존재하는 함수를 사용하거나 사용자가 정의한 람다 함수를 사용할 수도 있습니다.

```python
df.apply(np.cumsum)

df.apply(lambda x: x.max - x.min())
```

### 히스토그램 구하기 (Histogramming)

데이터의 값 빈도 조사

```python
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()
```

### 문자열 관련 메소드들 (String methods)

시리즈는 배열의 각 요소에 쉽게 적용이 가능하도록 str이라는 속성에 문자열을 처리할 수 있는 여러가지 메소드를 가지고 있습니다.

```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()
```

## 6. 합치기 (Merging)

다양한 정보를 담은 자료들이 있을 때 이들을 합쳐 새로운 자료를 만들어야 할 때가 있습니다

같은 형태의 자료들을 이어 하나로 만들어주는 `concat`, 다른 형태의 자료들을 한 컬럼을 기준으로 합치는 `merge`, 기존 데이터 프레임에 하나의 행을 추가하는 `append` 의 사용법에 대해 알아봅니다.

```python
pd.concat(df, df1)
# SQL스타일 합치기
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
merged = pd.merge(left, right, on='key')
#    key  lval  rval
# 0  foo     1     4
# 1  bar     2     5

#key값을 기준으로 조인
```



## 7. 묶기 (Grouping)

‘그룹화 (group by)’는 다음과 같은 처리를 하는 과정들을 지칭합니다.

- 어떠한 기준을 바탕으로 데이터를 나누는 일 (splitting)
  - 다중 컬럼 그룹핑도 가능 -> 계층구조의 인덱스,
- 각 그룹에 어떤 함수를 독립적으로 적용시키는 일 (applying)
- 적용되어 나온 결과들을 통합하는 일 (combining)

```python
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
df.groupby('A').sum() => #'A'로 그룹화 하고, 각 그룹에 함수를 독립적으로 적용
df.groupby(['A', 'B']).sum()
```



## 8. 변형하기 (Reshaping)

stack 메소드는 데이터 프레임의 컬럼들을 인덱스의 레벨로 만듭니다. 이를 '압축'한다고 표현합니다.

`df2`라는 데이터프레임은 A와 B컬럼을 가지고 있었지만 stack메소드를 통해 해 A 와 B 라는 값을 가지는 인덱스 레벨이 하나 더 추가된 형태로 변형되었습니다.

```python
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]

stacked = df2.stack()
# first  second   
# bar    one     A    0.029399
#                B   -0.542108
#        two     A    0.282696
#                B   -0.087302
# baz    one     A   -1.575170
#                B    1.771208
#        two     A    0.816482
#                B    1.100230
# dtype: float64

stacked.unstack(0) # 첫번째 수준부터 풀어주기
```

### Pivot Tables

```python
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
# 멀티 인덱스, 칼럼은 C의 값으로 구성, 필드 값은 'D'로 구성
```

