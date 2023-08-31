[toc]

# 텐서 플로우 tf.data.Dataset 사용 방법

기존에는 Tensorflow모델에 직접 feed_dict를 이용하여 값을 전달하였습니다. 하지만 이 방법은 batch 및 shiffle기능을 직접 구현해야 하며, 실제 모델이 학습하는데 느릴 수 있습니다. 텐서플로우 iris 튜토리얼에서도, tf.data.Dataset이라는 파이프라인을 이용하여 값을 입력합니다.

![img](https://hiseon.me/wp-content/uploads/2018/04/tensorflow-datasets.png)

tf.data.Dataset 모듈은 텐서플로우 Python Low Level API 와 상위의 High Level API의 Estimator 모듈 사이에 위치하는 Mid Level의 API입니다. Estimator 의 모델 입력에 사용될 수 있을 뿐만 아니라 직접적으로 데이터 참조에도 사용 될 수 있습니다. 그 아래는 아래의 이미지처럼 텐서플로우 커널이 존재하기 됩니다.

![img](https://hiseon.me/wp-content/uploads/2018/04/tensorflow-low-level-api.png)

## 데이터 파이프라인

> 데이터 파이프라인의 시작은 왜, 어디에서, 어떻게 데이터를 수집할 것인가에서 부터 시작한다.

데이터 파이프라인은 다양한 데이터 소스에서 원시 데이터를 수집한 다음 분석을 위해 데이터 레이크 또는 데이터 웨어하우스와 같은 데이터 저장소로 이전하는 방법입니다. 일반적으로 데이터는 데이터 저장소로 이동하기 전에 데이터 처리 과정을 거칩니다. 여기에는 적절한 데이터 통합과 표준화를 보장하는 필터링, 마스킹, 집계와 같은 데이터 변환이 포함됩니다. 이 과정은 데이터 세트의 대상이 관계형 데이터베이스인 경우 특히 중요합니다 .이 유형의 데이터 저장소에는 기존 데이터를 새 데이터로 업데이트하기 위한 정렬(즉, 데이터 열 및 유형 매칭)이 필요한 정의된 스키마가 있습니다.



데이터 파이프라인을 구축하기 위해서는 여러 소프트웨어적인 수동 작업들을 제거해야하며 Data가 각 지점을 순조롭게 흐르도록(flow)만들어야 한다. Data의 추출(extracting), 변경(transforming), 결합(combining), 검증(validating) 그리고 적재(loading)하는 과정들을 자동화 하는 것이다. 또한 여러 데이터 스트림을 한번에 처리해야 한다. 이 모든 과정은 오늘날 data-driven enterprise에서 필수적이다.

데이터파이프라인은 모든 종류의 스키마의 데이터를 수용해야한다. 입수하고자 하는 파일이 static source든 real time source이든 데이터파이프라인에서는 해당 파일의 데이터는 작은 단위(chnk)로 들어와서 병렬로 처리된다.

### 데이터 파이프라인과 ETL의 차이는?

아마 이 글을 보는 독자는 ETL이라는 단어를 들어봤을 것이다. ETL은 추출(Extract), 변환(Transform), 적재(Load)의 줄임이다. ETL시스템은 하나의 시스템에서 data를 추출하고, data를 변환하여 database나 data warehouse에 적재한다. 레거시 ETL 파이프라인은 보통 배치로 작동하고 큰 덩어리의 data를 특정 시간에 한 공간에 저장하는 작업을 한다. 예를 새벽 12:30에 시스템 트래픽이 낮아질 때 배치가 돌아서 데이터를 모아 적재하는 작업이 있을 수 있다.

 

반면에, **데이터 파이프라인은 ETL을 서브셋으로 포함하는 광범위한 용어**다. 데이터를 한 시스템에서 또다른 시스템으로 옮기는 작업을 뜻한다. 해당 데이터는 transform되는 경우도 있고 안하는 경우도 있다. 또한 실시간성으로 처리하는 것도 있고 배치성으로 처리할수도 있다. 데이터가 지속적으로 흘러서 업데이트되는 경우가 있는데 traffic 센서 모니터링과 같은 경우를 예로 들 수 있다. 데이터 파이프라인을 통해 가져온 데이터는 database나 data warehouse에 쌓지 않는 경우도 있고 혹은 다중으로 데이터를 쌓는 경우도 있다.

## 개요

텐서플로우 데이터셋 tf.data.Dataset은 아래와 같이 3가지 부분으로 나눠서 설명드리도록 하겠습니다.

- Dataset 생성 : tf.data.Dataset을 생성하는 것으로 메모리에 한번에 로드하여 사용할 수도 있으며, 동적으로 전달하여 사용할 수도 있습니다.
- Iterator 생성 : 데이터를 조회할때 사용되는 iterator 를 생성합니다.
- 데이터 사용 : 실제 모델에 데이터를 입력하거나, 읽게 됩니다.

## 텐서플로우 tf.data.Dataset 사용 방법

### Dataset 생성

아래와 같이 메모리에 로드된 데이터를 이용하여  Dataset을 생성할 수 있습니다.

```python
x= np.random.sample((10,2))
dataset = tf.data.Dataset.from_tensor_flow(x)
```

또한 데이터를 특성(feature)과 라벨(label)로 나누어 사용하는 경우처럼, 한 개 이상의 numpy 배열을 넣을 수도 있다.

```python
features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
```

#### tensor에서 불러오기

tensor를 사용해서 dataset을 초기화 할 수도 있다.

```python
# using a tensor
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))
```





