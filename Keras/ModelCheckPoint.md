[toc]

# **ModelCheckPoint로 모델 저장하기**

Early stopping 객체에 의해 트레이닝이 중지되었을 때, 그 상태는 이전 모델에 비해 일반적으로 validation error 가 높은 상태일 것이다. 따라서, Earlystopping 을 하는 것은 특정 시점에 모델의 트레이닝을 멈춤으로써, 모델의 validation error 가 더 이상 낮아지지 않도록 조절할 수는 있겠지만, 중지된 상태가 최고의 모델은 아닐 것이다. 따라서 가장 validation performance 가 좋은 모델을 저장하는 것이 필요한데, keras 에서는 이를 위해 ModelCheckpoint 라고 하는 객체를 존재한다. 이 객체는 validation error 를 모니터링하면서, 이전 epoch 에 비해 validation performance 가 좋은 경우, 무조건 이 때의 parameter 들을 저장한다. 이를 통해 트레이닝이 중지되었을 때, 가장 validation performance 가 높았던 모델을 반환할 수 있다. 

```python
# validation performance가 가장 최적일 때의 가중치를 저장할 수 있다.
mc = ModelCheckpoint('best_model.h5',monitor='val_loss', mode='min', save_best_only = True)
```

위 ModelCheckpoint instance를 callbacks 파라미터에 넣어줌으로써, 가장 validation performance 가 좋았던 모델을 저장할 수 있게된다.

```python 
hist = model.fit(train_x, train_y, nb_epoch=10, batch_size=10, verbose=2, validation_split=0.2, callbacks=[early_stopping, mc])  

```



```python
from keras.callbacks import ModelCheckpoint

tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch', options=None, **kwargs
)
```

`

## 인자설명

|         인자          |                             설명                             |
| :-------------------: | :----------------------------------------------------------: |
|     **filepath**      |             **모델을 저장할 경로를 입력합니다.**             |
|      **monitor**      | **모델을 저장할 때, 기준이 되는 값을 지정합니다.**<br>예를 들어, validation set의 loss가 가장 작을 때를 저장하고 싶으면 'val_loss'를 입력하고, 만약 train_set의 loss가 가장 작을 때 모델을 저장하고 싶으면 'loss'를 입력합니다. |
|      **verbose**      | **0,1** <br>1일 경우 모델이 저장 될 때, '저장되었습니다.'라고 화면에 표시되고, 0일 경우 화면에 표시되는 것 없이 바로 모델이 저장됩니다. |
|  **save_best_only**   | **True, False**<br />  True 인 경우, monitor 되고 있는 값을 기준으로 가장 좋은 값으로 모델이 저장됩니다. False인 경우, 매 에폭마다 모델이 filepath{epoch}으로 저장됩니다. (model0, model1, model2....) |
| **save_weights_only** | True, False  True인 경우, 모델의 weights만 저장됩니다. False인 경우, 모델 레이어 및 weights 모두 저장됩니다. |
|       **mode**        | **'auto', 'min', 'max'** <br /> val_acc 인 경우, 정확도이기 때문에 클수록 좋습니다. 따라서 이때는 max를 입력해줘야합니다. 만약 val_loss 인 경우, loss 값이기 때문에 값이 작을수록 좋습니다. 따라서 이때는 min을 입력해줘야합니다. auto로 할 경우, 모델이 알아서 min, max를 판단하여 모델을 저장합 |
|     **save_freq**     | 'epoch' 또는 integer(정수형 숫자) <br /> 'epoch'을 사용할 경우, 매 에폭마다 모델이 저장됩니다. integer을 사용할 경우, 숫자만큼의 배치를 진행되면 모델이 저장됩니다. 예를 들어 숫자 8을 입력하면, 8번째 배치가 train 된 이후, 16번째 배치가 train 된 이후 ..... 모델이 저장됩니다. |

