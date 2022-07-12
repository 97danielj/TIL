[toc]

# 4) 케라스의 SimpleRNN과 LSTM 이해하기

케라스의 SimpleRNN과 LSTM을 이해해봅니다.

# 1. 임의의 입력 생성하기

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional
```

우선 RNN과 LSTM을 테스트하기 위한 임의의 입력을 만듭니다.

```lua
train_X = [[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]
print(np.shape(train_X))
(4, 5)
```

위 입력은 단어 벡터의 차원은 5이고, 문장의 길이가 4인 경우를 가정한 입력입니다. 다시 말해 4번의 시점(timesteps)이 존재하고, 각 시점마다 5차원의 단어 벡터가 입력으로 사용됩니다. 그런데 앞서 RNN은 2D 텐서가 아니라 3D 텐서를 입력을 받는다고 언급한 바 있습니다. 즉, 위에서 만든 2D 텐서를 3D 텐서로 변경합니다. 이는 배치 크기 1을 추가해주므로서 해결합니다.

```python
train_X = [[[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1], [2.2, 1.4, 0.5, 0.9, 1.1]]] #RNN의 입력이다.
train_X = np.array(train_X, dtype=np.float32)
print(train_X.shape)
```

(batch_size, timesteps, input_dim)에 해당되는 (1, 4, 5)의 크기를 가지는 3D 텐서가 생성되었습니다. batch_size는 한 번에 RNN이 학습하는 데이터의 양을 의미하지만, 여기서는 샘플이 1개 밖에 없으므로 batch_size는 1입니다.

- SimpleRNN의 입력 차원 
  - 2D = input_shape(input_length, input_dim)
  - 3D = batch_input_shape(batch_size, input_length, input)
  
- SimpleRNN의 출력 차원 / 츨력의 크기는 은닉상태의 크기 $D_h$
  - 2D = output_shape(batch_size, hidden_units) / batch_size를 입력하지 않으면 ?
  - 3D =  output_shape(batch_size, timesteps,hidden_units)
  
  

# 2. SimpleRNN 이해하기

위에서 생성한 데이터를 SimpleRNN의 입력으로 사용하여 SimpleRNN의 출력값을 이해해보겠습니다. SimpleRNN에는 여러 인자가 있으며 대표적인 인자로 return_sequences와 return_state가 있습니다. 기본값으로는 둘 다 False로 지정되어져 있으므로 별도 지정을 하지 않을 경우에는 False로 처리됩니다. 우선, 은닉 상태의 크기를 3으로 지정하고, 두 인자 값이 모두 False일 때의 출력값을 보겠습니다.