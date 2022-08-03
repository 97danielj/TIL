[toc]

# Early Stopping의 개념과 Keras를 통한 구현

## Early Stopping 이란 무엇인가? 

딥러닝을 비롯한 머신러닝 모델의 한 가지 중요한 딜레마는 다음과 같다.

```tex
너무 많은 Epoch 은 overfitting 을 일으킨다. 하지만 너무 적은 Epoch 은 underfitting 을 일으킨다. 
```

이런 상황에서 Epoch을 어떻게 설정해야 하는가?

Epoch을 정하는데 많이 사용되는 **Early stopping**은 무조건 Epoch을 많이 돌린 후, 특정 시점에서 멈추는 것이다.

그 특정시점을 어떻게 정하느냐가 Early stopping 의 핵심이라고 할 수 있다. **일반적으로 hold-out validation set 에서의 성능이 더이상 증가하지 않을 때 학습을 중지시키게 된다.**  Keras 를 이용하여 Early stopping 을 구현하는 법과 성능이 더 이상 증가하지 않는다는 것은 어떤 기준으로 정하는 것인지를 중점으로 정리한다.

**모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수입니다.**

### Early Stopping in Keras

keras의 Early stopping을 구현하는 Early stopping함수를 통해 구현할 수 있다.

```python
from keras.callbacks import EarlyStopping
```



Early stopping 클래스의 구성요소

- **Performance measure : 어떤 성능을 monitoring할 것 인가?**
- **Trigger: 언제 training을 멈출 것인가?**

Early stopping 객체는 초기화 될 때 두개의 요소를 정의하게 된다.

아래와 같이 지정하면 validation set의 loss를 monitoring한다는 뜻이다.

```python
es = EarlyStopping(monitor = 'val_loss')
```

만약 performance measure가 최소화 시켜야하는 것이면 mode를 min 으로, 최대화 시켜야하는 것이면 mode를 max로 지정한다. loss 의 경우, 최소화 시키는 방향으로 training 이 진행되므로 min 을 지정한다. 

```python
es = EarlyStopping(monitor= 'val_loss', mode = 'min')
```

mode의 default는 auto인데, 이는 keras에서 알아서 min, max를 선택하게 된다. 여기까지가 가장 기본적인 Early Stopping의 사용법이다.

performace measure을 정의하고, 이것을 최대화 할지, 최소화 할지를 지정하는 것이다. 그러면 keras에서 알아서 적절한 epoch에서 training을 멈춘다. verbose=1로 지정하면, 언제 keras에서 training을 멈추었는지 화면에 출력할 수 있다.



성능이 증가하지 않는다고, 그 순간 바로 멈추는 것은 효과적이지 않을 수 있다. patience는 성능이 증가하지 않는 epoch을 몇 번이나 허용할 것인가를 정의한다. patience는 다소 주관적인 기준이다. 사용한 데이터와 모델의 설계에 따라 최적의 값이 바뀔수 잇다.

```python
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=510)
```



만약 performance measure 를 practical 하게 설정한 경우 성능의 증가의 기준을 직접 정의할 수 있다. 예를 들어 아래 코드는 validation accuracy 가 1% 증가하지 않는 경우, 성능의 증가가 없다고 정의한다. 
특정값에 도달했을 때, 더 이상 training 이 필요하지 않은 경우가 있다. 이 경우 baseline 파라미터를 통해 정의할 수 있다. 

```python
es = EarlyStopping(monitor = 'val_loss', mode='min', baseline=0.4)
```

최종적으로 model.fit 함수의 callback 으로 early stopping 객체를 넣어주면 early stopping 을 적용할 수 있다. 

```python
hist = model.fit(train_x, train_y, nb_epoch=10, batch_size=10, verbose=2, validation_split=0.2, callbacks=[early_stopping])
```



























