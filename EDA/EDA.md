[toc]

# EDA

EDA(탐색적 데이터 분석)

- **수집합 데이터가 들어왔을 때, 이를 다양한 각도에서 관찰하고 이해하는 과정**
- 데이터를 분석하기 전에 그래프나 통계적인 방법으로 자료를 직관적으로 바라보는 과정
- 데이터의 분포 및 값을 검토함으로써 데이터가 표현하는 현상을 더 잘 이해하고, **데이터에 대한 잠재적인 문제를 발견 가능**
- 데이터를 학습시키기위해 정리하는 과정
- 다양한 각도에서 살펴보는 과정을 통해 문제 정의 단계에서 미쳐 발생하지 못했을 다양한 패턴을 발견하고, 이를 바탕으로 기존의 가설을 수정하거나 새로운 가설을 세울 수 있다.



## 알고리즘을 이용한 성능 최적화

- 머신러닝과 딥러닝을 위한 알고림즘은 상당히 많음
- 수많은 알고리즘 중 우리가 선택한 알고리즘이 최적의 알고리즘 아닐 수도 있음
- 유사한 용도의 알고리즘들을 선택하여 모델을 훈련시켜보고 최적의 성능을 보이는 알고리즘을 선택해야함
- 머신러닝에서는 데이터 분류를 위해 SVM, K-최근접 이웃 알고리즘을 선택하여 훈련시켜보거나
- 시계열 데이터의 경우 RNN,LSTM,GRU 등의 알고리즘을 훈련시켜 성능이 가장 좋은 모델을 선택하여 사용



## 성능최적화

- 옵티마이저 및 손싷함수
  - 일반적으로 옵티마이저는 확률적 경사 하강법을 많이 사용
  - 네트워크 구성에따라 차이는 있지만 Adam이나 RMSProp등도 좋은 성능을 보이고 있음
  - 다양한 옵티마이저와 손실함수를 적용해보고 성능이 최고인 것을 선택
- 네트워크 구성
  - 네트워크 구성은 네트워크 토폴로지라고도 함
  - 최적의 네트워크를 구성하는 것 역기 쉽게 알 수 있는 부분이 아니기 때문에 네트워크 구성을 변경해가면서 성능을 테스트해야 함
  - 하나의 은닉층에 뉴런을 여러 개 포함시키거나(네트워크가 넓다고 표현),
  - 네트워크 계층을 늘리되 뉴런의 개수는 줄여 봄(네트워크가 깊다고 표현) - 데이터의 순수한 의미를 기억
  - 혹은 두 가지를 결합하는 방법으로 최적의 네트워크가 무엇인지 확인한 후 사용할 네트워크를 결정해야 함
- 하이퍼파라미터를 이용한 성능 최적화
  - 배치 정규화, 드롭아웃, 조기 종료가 있음
- 배치 정규화를 이용한 성능 최적화
- 정규화
  - 데이터 범위를 사용자가 원하는 범위로 제한하는 것을 의미
  - 각 특성 범위를 조정한다는 의미로 특성 스크레일링



## 1. 누락된 데이터 다루기

- 실제 애플리케이션에서는 훈련샘플에 하나 이상의 값이 누락된 경우가 드물지 않다.
- 수집과정에서 오류 , 측정방법의 적용불가로 생길수 있다.
- 일반적으로 누락된 값은 데이터 테이블에 빈 공간이나 예약된 문자열로 채워집니다.
- 숫자가 아니라는 의미의 NaN, 관계형 데이터베이스의 NULL
- 누락된 값을 다룰 수 없거나 단순히 이를 무시했을 때 예상치 못한 결과를 만듭니다.

### 1. 테이블 형태 데이터에서 누락된 값 식별

```python
df.isnull() # 셀이 수치 값을 가지는지 누락된 값을 가지는지 불리언 값 리턴

df.isnull().sum() #열마다 누락된 값의 개수를 리턴
# axis=0인 경우. 하나의 행 방향
# axis=1인 경우. 하나의 열 방향

df.values #데이터프레임을 넘파이 배열로 변환
```

### 2. 누락된 값이 있는 훈련 샘플이나 특성 제외

- 누락된 데이터를 다루는 가장 쉬운 방법 중 하나는 데이터셋에서 해당 훈련 샘픔(행)이나 특성(열)을 완전히 삭제하는 것입니다.
- 누락된 행은 dropna 메서드를 사용하여 쉽게 삭제할수 있습니다.

```python
df.dropna(axis=0) # 하나의 행 방향으로 삭제
df.dropna(axis=1) # 하나의 열 방향으로 삭제
df.dropna(how='all') #모든 행이 NaN일 때 행을 삭제

df.dropna(thresh=4) #NaN이 아닌 값이 4개 보다 작은 행을 삭제합니다.

df.dropna(subset=['C']) #특정 열에 NaN가 있는 행만 삭제합니다.
```

뉴락된 데이터를 제거하는 것이 간단해 보이지만 단점도 있습니다. 예를 들어 너무 많은 데이터를 제거하면 안정된 분석이 불가능 할 수 있습니다. 또는 너무 많은 특성 열을 제거하면 분류기가 클래스를 구분하는 데 필요한 중요한 정보를 잃을 위험이 있습니다. 



### 3. 누락된 값 대체

종종 훈련 샘플을 삭제하거나 특성 열을 통째로 제거하기 어려울 때가 있습니다. 유용한 데이터를 너무 많이 잃기 때문입니다. 이런 경우 여러가지 보간 기법을 사용하여 데이터셋에 있는 다른 훈련 샘플로부터 누락된 값을 추정할 수 있습니다.

1. 평균으로 대체

```python
#각 열의 평균으로 대체
from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr.fit(df.values)
imputed_data = imr.transform(df.values)

#각 행의 평균으로 대체
from sklearn.preprocessing import FunctionTransformer
ftr_imr = FunctionTransformer(lambda X:imr.fit_transform(X.T).T, validate=False)
imputed_data = ftr_imr.transform(df.values)
imputed_data

# 가장 많이 사용되는 보간 메소드
df.fil
```

### 4. 사이킷런 추정기 API

SimpleImputer 클래스를 사용하여 데이터셋에 있는 누락된 값을 대체했습니다. SimpleImputer클래스는 데이터 변환에 사용되는 사이킷런의 `변환기`클래스입니다. 이런 추정기의 주요 메소드 두 개는 fit와 transform입니다. `fit메서드`를 사용하여 훈련 데이터에서 모델 파라미터를 학습합니다. `transform메서드`를 사용하여 학습한 파라미터로 데이터를 변환합니다. 변환하려는 데이터 배열은 모델 학습에 사용한 데이터의 특성 개수와 같아야 합니다.

변환기 클래스와 개념상 매우 유사한 API를 가진 사이컷런의 `추정기(estimator)` 입니다. 추정기는 predict메소드가 있지만 transform메서드도 가질 수 있습니다. 분류를 위한 추정기를 훈련할 때 fit메서드를 사용해서 모델의 파라미터를 학습했습니다. 지도 학습 작업에서는 모델 훈련 시 추가적인 클래스 레이블을 제공. 그런 다음 predict메서드를 사용하여 레이블이 없는 새로운 데이터 샘플에대한 예측을 만듭니다.

## 2. 범주형데이터 다루기

### 1. 범주형 데이터 인코딩

범주형 데이터 중에서는 `순서가 있는 데이터 특성`(옷 사이즈), `순서가 없는 데이터 특성`이 있을 수 있다.

### 2. 순서가 있는 특성 매핑

학습 알고리즘이 순서 특성(범주형 순서 데이터)을 올바르게 인식하려면 **범주형의 문자열 값을 정수**로 바꾸어야 합니다. 따로 사용자가 특성의 순서를 올바르게 변환시키는 매핑함수를 직접 만들어야 합니다. 만약 특성간 산술적인 차이를 알고있다면 매핑함수를 구현하기 쉽습니다.
$$
XL=L+1 = M+2
$$
size특성에서 산술적인 차이를 이용한 매핑

```python
size_mapping = {
    'XL':3,
    'L':2,
    'M':1
}

df['size'] = df['size'].map(size_mapping)
df
```

### 3. 클래스 레이블 인코딩

- **클래스 레이블은 순서가 없다는 것을 기억**

- 많은 머신 러닝 라이브러리는 클래스 레이블이 정수로 인코딩되었을 것이라고 기대합니다.

- 사이킷런의 분류 추정기(estimator) 대부분은 자체적으로 클래스 레이블을 정수로 변환해주지만 사소한 실수를 방지하기 위해 클래스 레이블을 정수 배열로 전달하는 것이 좋은 습관입니다.

- 특정 문자열 레이블에 할당한 정수는 아무런 의미가 없다.

- enumerate를 사용하여 클래스 레이블을 0 부터 할당합니다.

- ```python
  import numpy as np
  class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
  class_mapping
  
  df['classlabel'] = df['classlabel'].map(classlabel)
  
  ```

- 키-값 쌍을 뒤집어서 변환된 클래스 레이블을 다시 원본 문자열로 바꿀 수 있다.

- 다른 방법으로 LabelEncoder클래스를 사용하면 편리합니다.

- ```python
  from sklearn.preprocessing import LabelEncoder
  class_le = LabelEncoder()
  y = class_le.fit_transform(Df['classlabel'].values)
  y
  # array([0, 1, 0])
  
  #y = class_le.classes_
  #array(['class1', 'class2'], dtype=object) 각 클래스레이블이 저장되어 있어서
  #inverse_transform()도 가능하다.
  ```

- 여러가지 범주형 데이터 열을 정수로 변환

- ```python
  from sklearn.compose import ColumnTransformer #트랜스포머(변환기)
  from sklearn.preprocessing import OrdinalEncoder #인코더
  ord_enc = OrdinalEncoder(dtype=int)# 변환기 객체
  col_trans=ColumnTransformer([('ord_enc',ord_enc,['color'])])
  #첫 번째 매개변수 : 트랜스포머 리스트[('이름',변환기객체,변환할 열)]
  X_trans = col_trans.fit_transform(df)
  X_trans
  
  '''
  OrdinaryEncoder클래스 dtype매개변수 기본값 np.float64
  '''
  ```

### 4. 순서가 없는 원-핫 이코딩 적용

- 범주형에서 정수로 인코딩 시 클래스마다 부여된 정수에는 아무 의미가 없지만 학습 알고리즘은 green은 blue보다 크고 red는 green보다 크다고 가정할 것입니다.
- 이 문제를 해결하기 위한 원 핫 인코딩 기법
- 순서 없는 특성에 들어 있는 고유 값마다 새로운 더미 특성을 만드는 것입니다.
- 예를 들어 Color특성을  blue, green, red 고유 값으로고 구성된 특성으로 변환시키는것.
- 이진 값을 사용해서 이다/아니다로 표현

```python
from sklearn.preprocessing import OneHotEncoder
X=df[['color','size','price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray()
```

- ColumnTransformer :  여러 특성을 정수로 변환시키는 변환기

```python
from sklearn.compose import ColumnTransformer
X = df[['color','size','price']].values
c_transfer = ColumnTransformer([('onehot',OrdinalEncoder(dtype=int),[0]),
                               ('nothing','passthrough',[1,2])
                               ])
```

- 원-핫 인코딩의 더 쉬운 방식 : get_dummies(df[]) -> 문자열 열만 더미 특성으로 변경

```python
pd.get_dummies(df[['price','color','size']])
#문자열 특성만 원-핫 인코딩 한다.
#columns 매개변수 사용 시 더미 특성으로 변환할 컬럼 지정
```

- ColumnTransformer : 여러 특성을 정수로 변환시키는 변환기

```python
from sklearn.compose import ColumnTransformer
X = df[['color','size','price']].values
c_transf = ColumnTransformer([('onehot',OrdinalEncoder(dtype=int),[0]),
                               ('nothing','passthrough',[1,2])
                               ])
# 첫 번째 문자열을 인코딩
```

- **원-핫 인코딩된 데이터셋을 사용할 때 다중 공선성 문제가 발생 할 수 있다.** 특성 간의 상관관계가 높으면 역행렬을 계싼하기 어려워 수치적으로 불안정해집니다. 변수 간의 상관관계를 감소하려면 원-핫 인코딩된 배열에서 특성 열 하나를 삭제합니다. 특성 하나를 삭제하더라도 일는 정보가 없다. 모두 0일때가 삭제된 열의 고유 값을 나타냅니다.

```python
pd.get_dummied(df[['price','color','size']],drop-first=True)
color_ohe = OneHotEncoder(categories='auto', drop='first')
#범주형 데이터 자동으로 찾고, 원핫인코딩-> 첫번째 열 삭제
c_transf = ColumnTransformer([
    ('onehot',color_ohe,[0]),
    ('nothing','passthrough',[1,2])
])
c_transf.fit_transform(X)
```



## 3. 데이터셋 분할

모델을 실전에 투입 전에 테스트 데이터셋에 있는 레이블과 예측을 비교합니다.

매개변수 stratify =y : 훈련데이터셋과 테스트데이터셋에 원본 데이터셋의 레이블 클래스 비율을 동일하게 유지 시킨다.



## 4. 특성 스케일 맞추기

특성 스케일 조정은 전처리 파이프라인에서 잊어버리기 쉽지만 아주 중요한 단계입니다. 결정 트리와 랜덤포레스트는 특성 스케일 조정에 대해 걱정할 필요가 없는 몇 안되는 머신러닝 알고리즘 중 하나입니다.

- `경사하강법` 알고리즘을 구현하면서 보았듯이 대부분의 머신 러닝과 최적화 알고리즘은 특성의 스케일이 같을 때 훨씬 성능이 좋습니다.

- 스케일이 같지 않다면 아달린에서 가중치 최적화는 스케일이 큰 특성에 맞추워 가중치를 최적화 할 것입니다.

- 정규화(normalizaton) 

  - 특성의 스케일을 [0,1]에 맞추는 것. 최소-최대 스케일 변환
  - 정해진 범위의 값이 필요할 때 유용하게 사용할 수 있는 기법

- 다양한 스케일러

  - RoubustScaler : 중간값은 빼고, 1분위수와 3분위수를 사용해서 데이터셋을 조정하므로 극단적인 값과 이상치에 영향을 덜 받습니다.

    - 이상치가 많이 포함된 작은 데이터셋을 다룰 때 특히 도움이 된다.

    - $$
      x_{robust}^{(i)} = \dfrac{x^{(i)} - q_2}{q_3 - q_1}
      $$

  - MinMaxScaler : 최대값을 1, 최소값을 0으로 맞춰 특성 열 변환
  - StandardScaler : 평균 0, 표준편차 1로 초기화
  - MaxAbsScaler : 배열을 절대값 중 최대값인 원소로 나눈다. 최대값은 1이 된다.

  ```python
  from sklearn.preprocessing import MinMaxScaler
  #최소-최대 스케일 변환
  #이상치에 민감
  from sklearn.preprocessing import RobustScaler
  ```

- 표준화(standardization)

  - 특성의 평균을 0에 맞추고, 표준편차를 1로 만들어 표준 정규 분포와 같은 특징을 가지도록 만든다.
  - 머신러닝 알고리즘, 경사하강법같은 최적화 알고리즘에서 유용하게 사용된다.

  ```python
  from sklearn.preprocessing import StandardScaler
  ```


## 5. 유용한 특성 선택

모델이 테스트 데이터셋보다 훈련 데이터셋에서 성능이 훨씬 높다면 과대적합에 대한 강력한 신호입니다. `과대적합`은 모델 파라미터가 훈련 데이터셋에 있는 특정 샘플들에 대해 너무 가깝게 맞추어져 있다는 의미 입니다. 새로운 데이터를 일반화 화지 못하기 때문에 모델 분산이 크다고 합니다. `과대적합`의 주된 원인은 주어진 훈련데이터에 비해 모델이 너무 복잡하기 때문입니다.

일반화 오차를 감소시키기 위해 사용하는 방식

- 더 많은 훈련데이터를 모읍니다.
- 규제를 통해 복잡도를 제한합니다.
- 파라미터 개수가 적은 간단한 모델을 선택
- 데이터 차원을 줄입니다.

### 1. 모델 복잡도 제한을 위한 L1과 L2규제

- L2 규제

  - 개별 가중치 값을 제한하여 모델 복잡도를 줄이는 한 방법

  - $$
    \lVert \boldsymbol{x} \rVert_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}
    $$

  - ```python
    l2_norm = np.sqrt(np.sum(ex_2f ** 2, axis=1))
    print(l2_norm)
    ex_2f / l2_norm.reshape(-1, 1)
    ```

  - 샘플 별로 특성의 제곱을 더하고, 이 값의 제곱근을 구하면 L2노름 입니다. 그다음 각 샘플의 특성을 해당 L2노름으로 나눕니다.

- L1 규제

  - 개별 가중치 절대값을 더한 것이 L1노름입니다.

  - $$
    \lVert \boldsymbol{x} \rVert_1 = \lvert x_1 \rvert + \lvert x_2 \rvert + \cdots + \lvert x_n \rvert
    $$

  - 대부분의 특성 가중치가 0이됩니다.

  - 보통 희소한 특성 벡터를 만든다.

  - 실제로 관련없는 특성이 많은 고차원 데이터셋일 경우 이런 희소성이 도움이 될 수 있습니다.

### 2. L2 규제의 기하학적 해석

![](https://git.io/JtY8I)

- L2규제는 비용함수에 패널티 항을 추가합니다.
- 규제가 없는 비용함수로 훈련한 모델에 비해 가중치 값을 아주 작게 만드는 효과가 있습니다.
- 우리의 목표는 훈련데이터에서 비용함수를 최소화하는 가중치 값의 조합을 찾는 것
- 규제를 더 작은 가중치를 얻기 위해 비용함수에 추가하는 패널티 항으로 생각할 수 있습니다.
- 규제 파라미터로 규제의 강도를 크게하면 가중치가 0에 가까워지고 훈련데이터에 대한 모델 의존성은 줄어듭니다.

![](https://git.io/JtY8L)

- 가중치 값은 규제 예산을 초과할 수 없습니다. 
- 패널티 제약이 있는 상황에서 최선은 L2회색공과 규제가 없는 비용함수의 등고선이 만나는 지점입니다.
- 우리의 목표는 규제가 없는 비용과 패널티 항의 합을 최소화하는 것입니다.
- 학습 데이터가 충분치 않을때 편향을 추가하여 모델을 간단하게 만듦으로써 분산을 줄이는 것으로 해석

### 3. L1 규제를 사용한 희소성

L1 패널티는 가중치의 절대값의 합이기 때문에 다이아몬드 모양의 제한 범위를 그릴 수 있습니다.

![](https://git.io/JtY8t)



### 4. 순차 특성 선택 알고리즘

모델 복잡도를 줄이고 과대적합을 피하는 다른 방법은 특성 선택을 통한 `차원 축소`입니다. 규제가 없는 모델에서 특히 유용합니다.(랜덤 포레스트, 결정 트리)

차원 축소 기법에는 `특성선택`과 `특성추출`이 있습니다. 특성 선택은 원본상에서 일부를 선택합니다. 특성 추출은 일련의 특성에 얻은 정보로 새로운 특성을 만듭니다.

- **순차 특성 선택**
  - 탐욕적 탐색 알고리즘으로 초기 d차원의 특성 공간을 k<d k차원으로 특성 부분 공간으로 축소합니다.
  - 특성 선택 알고리즘은 주어진 문제에 가장 관련이 높은 특성 부분 집합을 자동으로 선택하는 것이 목적입니다.
  - 관계없는 특성이나 잡음을 제거하여 계산 효율성을 높이고 모델의 일반화 오차를 줄입니다.
  - 규제를 제공하지 않는 알고리즘을 사용할 때 유용합니다.
  - **순차 후진 선택** : 계산 효울성을 향상하기 위해 모델 성능을 가능한 적게 희생하면서 초기 특성의 부분 공간으로 차원을 축소합니다.

## 6. 랜덤 포레스트의 특성 중요도 사용

랜덤 포레스트를 사용하면 앙상블에 참여한 모든 결정 트리에서 계산한 평균적인 불순도 감소로 **특성 중요도를 측정할 수 있습니다.** 

```python
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
importances #각 특성마다 중요도 값 리스트. 정규화된 값
```



- 랜덤 포레스트에서 두 개 이상의 특성이 매우 상관관계가 높다면 하나의 특성은 매우 높은 순위를 갖니만 다른 특성 정보는 완전히 잡아내지 못할 수 있습니다. 특성 중요도 값을 해석하는 것보다 모델의 예측성능에만 관심이 있다면 이 문제를 신경 쓸 필요는 없다.

