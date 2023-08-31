[toc]

# 08) 케라스의 함수형 API(Keras Functional API)

앞서 구현한 선형, 로지스틱, 소프트맥스 회귀 모델들과 케라스 훑어보기 실습에서 배운 케라스의 모델 설계 방식은 Sequential API을 사용한 것입니다. 그런데 Sequential API는 여러층을 공유하거나 다양한 종류의 입력과 출력을 사용하는 등의 복잡한 모델을 만드는 일에는 한계가 있습니다. 이번에는 더욱 복잡한 모델을 생성할 수 있는 방식인 Functional API(함수형 API)에 대해서 알아봅니다.

Functional API에 대한 자세한 소개는 케라스 공식 문서에서도 확인할 수 있습니다.

## **1. Sequential API로 만든 모델**

두 가지 API의 차이를 이해하기 위해서 앞서 배운 Sequential API를 사용하여 기본적인 모델을 만들어봅시다.

```javascript
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
```

위와 같은 방식은 직관적이고 편리하지만 단순히 층을 쌓는 것만으로는 구현할 수 없는 복잡한 신경망을 구현할 수 없습니다. 따라서 초심자에게 적합한 API이지만, 전문가가 되기 위해서는 결과적으로 Functional API를 학습해야 합니다.



## **2. Functional API로 만든 모델**

**Functional API는 각 층을 일종의 함수(function)로서 정의합니다. 그리고 각 함수를 조합하기 위한 연산자들을 제공하는데, 이를 이용하여 신경망을 설계**합니다. Functional API로 FFNN, RNN 등 다양한 모델을 만들면서 기존의 sequential API와의 차이를 이해해봅시다.

- Functional API는 입력의 크기(shape)를 명시한 입력층(Input layer)을 모델의 앞단에 정의해주어야 합니다.

### **1) 전결합 피드 포워드 신경망(Fully-connected FFNN)**

Sequential API와는 다르게 functional API에서는 입력 데이터의 크기(shape)를 인자로 **입력층을 정의**해주어야 합니다. 피드 포워드 신경망(Fully-connected FFNN)을 만든다고 가정해보겠습니다.

``` python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
```

```python
inputs = Input(shape=(10,))
```

위의 코드는 10개의 입력을 받는 입력층을 보여줍니다. 위의 코드에 은닉층과 출력층을 추가해봅시다.

```python
inputs = Input(shape=(10,))
hidden1 = Dense(64, activation='relu')(inputs)  # <- 새로 추가
hidden2 = Dense(64, activation = 'relu')(hidden1)  # <- 새로 추가
output = Dense(1, activation='sigmoid')(hidden2)  # <- 새로 추가
```

위의 코드를 하나의 모델로 구성해보겠습니다. 이는 Model에 입력 텐서와 출력 텐서를 정의하여 완성됩니다.

```python
model = Model(inputs=inputs, outputs=output) # <- 새로 추가
```

지금까지의 내용을 정리하면 다음과 같습니다.

- Input() 함수에 입력의 크기를 정의합니다.
- **이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당합니다**.
- Model() 함수에 입력과 출력을 정의합니다.
- 이를 model로 저장하면 sequential API를 사용할 때와 마찬가지로 model.compile, model.fit 등을 사용 가능합니다.

```python
odel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(data, labels)
```



이번에는 위에서 배운 내용을 바탕으로 선형 회귀와 로지스틱 회귀를 Functional API로 구현해봅시다.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import model

X = [1,2,3,4,5,6,7,8,9] #공부하는 시간
y = [11,22,33,44,53,66,55,87,95] # 각 공부하는 시간에 맵핑되는 성적

inputs = Input(shape(1,))
output = Dense(1,activation='linear')(inputs)
linear_model = Model(inputs, output)
sgd = optimizers.SGD(lr=0.01)

linear_model.compile(optimizer = sgd, loss='mse', metrics=['mse'])
linear_model.fit(X,y,epochs=300)


```

그 외에 다양한 다른 예제들을 구현해봅시다.

### 3) 로지스틱 회기

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Model
inputs = Input(shape=(3,))
output = Dense(1, activation='sigmoid')(inputs)
logistic_model = Model(inputs, output)
```



### 4) 다중 입력을 받는 모델

functional API를 사용하면 아래와 같이 다중 입력과 다중 출력을 가지는 모델도 만들 수 있습니다.

```python
# 최종 완성된 다중 입력, 다중 출력 모델의 예
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
```



```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# 두 개의 입력층을 정의
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

# 첫번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
x = Dense(16, activation="relu")(inputA)
x = Dense(8, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# 두번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(8, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# 두개의 인공 신경망의 출력을 연결(concatenate)
result = concatenate([x.output, y.output])

z = Dense(2, activation="relu")(result)
z = Dense(1, activation="linear")(z)

model = Model(inputs=[x.input, y.input], outputs=z)
```



### 5) RNN 은닉층 사용하기

이번에는 RNN 은닉층을 가지는 모델을 설계해봅시다. 여기서는 하나의 특성(feature)에 50개의 시점(time-step)을 입력으로 받는 모델을 설계해보겠습니다. RNN에 대한 구체적인 내용은 다음 챕터인 RNN 챕터에서 배웁니다.

```python
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

inputs = Input(shape=(50,1))
lstm_layer = LSTM(10)(inputs)
x = Dense(10, activation='relu')(lstm_layer)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs = ouput)

```

### **6) 다르게 보이지만 동일한 표기**

케라스의 Functional API가 익숙하지 않은 상태에서 Functional API를 사용한 코드를 보다가 혼동할 수 있는 점이 한 가지 있습니다. 바로 동일한 의미를 가지지만, 하나의 줄로 표현할 수 있는 코드를 두 개의 줄로 표현한 경우입니다.

```python
result = Dense(128)(input)
```

위 코드는 아래와 같이 두 개의 줄로 표현할 수 있습니다.

```python
dense = Dense(128)
result = dense(input)
```