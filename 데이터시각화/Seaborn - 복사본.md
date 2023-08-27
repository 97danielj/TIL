[toc]

# Seaborn을 사용한 데이터 분포 시각화

seaborn은 Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지 이다.

## 1. 1차원 분포 플롯

**1차원 데이터는 실수 값이면 히스토그램과 같은 실수 분포 플롯으로 나타내고 카테고리 값이면 카운트 플롯으로 나타낸다.**

```python
iris = sns.load_dataset("iris")    # 붓꽃 데이터
titanic = sns.load_dataset("titanic")    # 타이타닉호 데이터
tips = sns.load_dataset("tips")    # 팁 데이터
flights = sns.load_dataset("flights")    # 여객운송 데이터
```



### 1. 1차원 실수 분포 플롯

실수 분포 플롯은 자료의 분포를 묘사하기 위한 것으로 Matplotlib의 단순한 히스토그램과 달리 커널 밀도 및 러그 표시 기능 및 다차원 복합 분포기능을 제공한다.

- **1차원 실수 분포 플롯 명령에는 rugplot, kdeplotm distplot**
- 러그(rug)플롯은 데이터 위치를 x축 위에 작은 선분(rug)으로 나타내어 **실제 데이터들의 위치를 보여준다.**

```python
x = iris.petal_length.values
sns.rugplot(x)
plt.title("Iris 데이터 중, 꽃잎의 길이에 대한 Rug Plot ")
plt.show()
```

- 커널 밀도: 커널이라는 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포 곡선을 보여주는 방법이다.

```python
sns.kdeplot(x) #커널 밀도 분포 그래프
plt.title("Iris 데이터 중, 꽃잎의 길이에 대한 Kernel Density Plot")
plt.show()
```

- **distplot : 러그와 커널 밀도 표시기능 + 히스토그램 **
  - 데이터의 실제위치
  - 데이터의 분포형태
  - 데이터의 계급별 분포표

```python
sns.distplot(x, kde=True, rug=True)
plt.title("Iris 데이터 중, 꽃잎의 길이에 대한 Dist Plot")
plt.show()
```





### 2. 1차원 카테고리 분포 플롯

**살제데이터값들이 카테고리값(범주형)이라면 카운트플롯을 사용한다.**

- countplot : 카테고리 값별로 데이터가 얼마나 있는지 표시할 수 있다.

- `countplot` 명령은 데이터프레임에만 사용할 수 있다. 사용 방법은 다음과 같다.

- ```python
  countplot(x="column_name", data=dataframe)
  #x에는 데이터프레임의 열 이름 문자열을 넣는다.
  
  sns.countplot(x='day', data=tips)
  plt.title('요일별 팁을 준 횟수')
  plt.show()
  
  sns.countplot(x='class',data=titanic)
  plt.title('타이타닉호 각 클래스별 승객수')
  plt.show()
  ```



## 2. 다차원 데이터

**데이터 변수가 여러개인 다차원 데이터**는 데이터의 종류에 따라사 다음과 같은 경우가 있을 수 있습니다.

- 분석하고자 하는 데이터가 모두 실수 값인 경우
- 분석하고자 하는 데이터가 모두 카테고리 값인 경우
- 분석하고 데이터가 모두 실수 값과 카테고리 값이 섞어 있는 경우

### 1.  실수형 데이터

#### 2차원 실수형 데이터

만약 데이터가 2차원이고 모두 연속적인 실수값이라면 스캐터 플롯(scatter plot)을 사용하면 된다. 스캐터 플롯을 그리기 위해서는 Seaborn패키지의 jointplot명령을 사용한다. 

- **jointplot : 스캐터 플롯뿐 아니라 차트의 가장자리에 각 변수의 히스토그램도 그린다.**

- `jointplot` 명령도 데이터프레임에만 사용할 수 있다. 사용 방법은 다음과 같다.

- ```python
  jointplot(x="x_name", y="y_name", data=dataframe, kind='scatter')
  #data : 데이터프레임
  #x: x축 변수가 될 df의 열 이름 문자열
  #y: y축 변수가 될 df의 열 이름 문자열
  #kind : 차트의 종류  ='scatter'이면 스캐터 플롯
  
  sns.jointplot(x='sepal_length', y='sepal_width', data=iris)
  plt.suptitle("꽃받침의 길이와 넓이의 Joint Plot", y=1.02)
  plt.show()
  
  sns.jointplot(x="sepal_length", y="sepal_width", data=iris, kind="kde") #커널밀도함수(연속형 데이터 히스토그램)
  plt.suptitle("꽃받침의 길이와 넓이의 Joint Plot 과 Kernel Density Plot", y=1.02)
  plt.show()
  ```



#### 다차원 실수형 데이터

만약 3차원 이상의 데이터라면 seaborn 패키지의 pairplot명령을 사용한다**. pairplot은 데이터프레임을 인수로 받아 그리드 형태로 각 데이터 열의 조합에 대해 스캐터 플롯**을 그린다. **같은 데이터가 만나는 대각선 영역에는 해당 데이터의 히스토그램을 그린다.**

```python
sns.pairplot(iris)
plt.title("Iris Data의 Pair Plot")
plt.show()
```

만약 카테고리형 데이터가 섞어 있는 경우에는 hue인수에 카테고리 변수 이름을 지정하여 카테고리 값에 따라 색상을 다르게 할 수 있다.

```python
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
#데이터프레임에 카테고리형 데이터가 있는 경우
#hue인수에 카테고리 변수를 지정해서 색상을 다르게 할 수 있다.
plt.title("Iris Pair Plot, Hue로 꽃의 종을 시각화")
plt.show()
```



### 2. 카테고리 데이터
#### 2차원 카테고리 데이터

만약 데이터가 2차원이고 모든 값이 카테고리 값이면 heatmap명령을 사용한다.

```python
titanic_size = titanic.pivot_table(index='class',columns='sex', aggfunc = 'size')
#행인덱스 : class
#열인덱스: sex
#집계함수 : size
```

```python
sns.heatmap(titanic_size, cmap = sns.light_palette("gray", as_cmap=True),
annot=True, fmt='d')
plt.title('Heatmap')
plt.show()
```



### 3. 복합데이터

#### 1. 2차원 복합데이터

만약 데이터가 2차원이고 실수 값, 카테고리 값이 섞여 있다면 기존의 플롯 이외에도 다음과 같은 분포 플롯들을 이용할 수 있다. => 기존 pairplot(hue사용)

- `barplot`

  - 카테고리 값에 따른 실수 값의 평균과 편차를 표시하는 기본적인 바 차트를 생성.

  - 평균은 막대의 높이, 편차는 에러바

  - ```python
    sns.barplot(x="day", y="total_bill", data=tips)
    plt.title("요일 별, 전체 팁")
    plt.show()
    ```

- `boxplot`

  - 박스 플롯은  박스와 박스 바깥의 선으로 이루어진다.

  - 박스는 실수 값 분포에서 1사분위수(Q1) 와 3사분위수(Q3)를 뜻하고  Q3-Q1를 IQR이라고 한다. 박스 내부 가로선은 중앙값. 박스 외부의 의 세로선은 1사분위 수보다 1.5 x IQR 만큼 낮은 값과 3사분위 수보다 1.5 x IQR 만큼 높은 값의 구간을 기준으로 그 구간의 내부에 있는 가장 큰 데이터와 가장 작은 데이터를 잇는 선분이다. 그 바깥의 점은 아웃라이어

  - ```python
    sns.boxplot(x="day", y="total_bill", data=tips)
    plt.title("요일 별 전체 팁의 Box Plot")
    plt.show()
    ```

  - `boxplot`이 중앙값, 표준 편차 등, 분포의 간략한 특성만 보여주는데 반해 `violinplot`, `stripplot`. `swarmplot` 등은 카테고리값에 따른 각 분포의 실제 데이터나 전체 형상을 보여준다는 장점이 있다.

- `violinplot`

  - 세로 방향으로 커널 밀도 히스토그램을 그려주는데 왼쪽과 오른쪽이 대칭되도록 하여 바이올린이 처럼 보인다.

  - ```python
    sns.violinplot(x="day", y="total_bill", data=tips)
    plt.title("요일 별 전체 팁의 Violin Plot")
    plt.show()
    ```

  

- `stripplot`

  - `stripplot`은 마치 스캐터 플롯처럼 모든 데이터를 점으로 그려준다.

  -  `jitter=True`를 설정하면 가로축상의 위치를 무작위로 바꾸어서 데이터의 수가 많을 경우에 겹치지 않도록 한다.

  - ```python
    import numpy as np
    
    np.random.seed(0)
    sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
    #x는 가로축 카테고리 데이터 열
    #y는 세로축 실수 데이터 열
    #jiter은 가로축상의 위치를 무작위로 바꾸어서 데이터의 수가 많을 경우 겹치지 않도록 한다.
    #스캐터 플롯을 그려준다.
    plt.title("요일 별 전체 팁의 Strip Plot")
    plt.show()
    ```



#### 2. 다차원 복합데이터

위에서 배운 명령어에는 2차원 이상의 고차원 데이터에 대해서도 분석할 수 있는 기능이 포함되어있다.

barplot,violinplot, boxplot등에서의 두 가지 카테고리 값에 의한 실수 값의 변화를 보기 위한  hue인수에 카테고리 값을 가지는 변수의 이름을 지정하면 카테고리 값에 따라 다르게 시각화된다.

- barplot : 카테고리 값에 따른 실수값의 평균과 표준편차

```python
sns.barplot(x="day", y="total_bill", hue="sex", data=tips)
plt.title("요일 별, 성별 전체 팁의 Histogram")
plt.show()
```

- boxplot :  실수값 분포의 1사분위수(Q1) 와 3사분위수(Q3)를 뜻한다.

- violinplot : 세로방향의로 실수값의 커널 밀도 함수를 보여준다.

```python
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips)
#split=True 적용시 hue에속한 카테고리를 하나로 합쳐주고 커널 밀도 함수를 좌, 우 하나씩 맡게 한다.
plt.title("요일 별, 성별 전체 팁의 Violin Plot")
plt.show()
```

- stripplot : 카테고리별 실수값들의 산점도를 보여준다.

```python
np.random.seed(0)
sns.stripplot(x="day", y="total_bill", hue="sex", data=tips, jitter=True)
#split = true : heu에 속한 카테고리도 서로 나누어 준다.
plt.title("요일 별, 성별 전체 팁의 Strip Plot")
plt.legend(loc=1)
plt.show()
```

- swarmplot : stripplot과 매우 유사하나 같은 카테고리('x')에 속하는 데이터들이 안겹치게 그려준다.

```python
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)
#split = true : heu에 속한 카테고리도 서로 나누어 준다.
plt.title("요일 별, 성별 전체 팁의 Swarm Plot")
plt.legend(loc=1)
plt.show()
```

- heatamap : 두 개의 카테고리 값에 의한 실수 값 변화를 볼수 있다.

```python
flights_passengers = flights.pivot("month", "year", "passengers")
plt.title("연도, 월 별 승객수에 대한 Heatmap")
sns.heatmap(flights_passengers, annot=True, fmt="d", linewidths=1)
plt.show()
#heatmap은 colormap으로 표시 해준다. 밝을수록 해당데이터 size가 많은것
```









