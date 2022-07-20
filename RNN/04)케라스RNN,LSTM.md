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

```tex
(1, 4, 5)
```

(batch_size, timesteps, input_dim)에 해당되는 (1, 4, 5)의 크기를 가지는 3D 텐서가 생성되었습니다. batch_size는 한 번에 RNN이 학습하는 데이터의 양을 의미하지만, 여기서는 샘플이 1개 밖에 없으므로 batch_size는 1입니다.

- SimpleRNN의 입력 차원 
  - 2D = input_shape(input_length, input_dim)
  - 3D = batch_input_shape(batch_size, input_length, input)
  
- SimpleRNN의 출력 차원 / 츨력의 크기는 은닉상태의 크기 $D_h$
  - return_sequences = False인 경우 : 2D = output_shape(batch_size, hidden_units) 
  - return_sequences = True인 경우 : 3D =  output_shape(batch_size, timesteps,hidden_units)
  
  

# 2. SimpleRNN 이해하기

위에서 생성한 데이터를 SimpleRNN의 입력으로 사용하여 SimpleRNN의 출력값을 이해해보겠습니다. SimpleRNN에는 여러 인자가 있으며 대표적인 인자로 return_sequences와 return_state가 있습니다. 기본값으로는 둘 다 False로 지정되어져 있으므로 별도 지정을 하지 않을 경우에는 False로 처리됩니다. 우선, 은닉 상태의 크기를 3으로 지정하고, 두 인자 값이 모두 False일 때의 출력값을 보겠습니다.

앞으로의 실습에서 SimpleRNN을 매번 재선언하므로 은닉 상태의 값 자체는 매번 초기화되어 이전 출력과 값의 일관성은 없습니다. 그래서 출력값 자체보다는 해당 값의 크기(shape)에 주목해야합니다.

```python
rnn = SimpleRNN(3)
# rnn = SimpleRNN(3, return_sequences=False, return_state=False)와 동일.
hidden_state = rnn(train_X)
#(1, 4, 5)
print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
```

```tex
hidden state : [[ 0.99989235 -0.9704083  -0.6916766 ]], shape: (1, 3)
```

(1, 3) 크기의 텐서가 출력되는데, 이는 마지막 시점의 은닉 상태입니다. 은닉 상태의 크기를 3으로 지정했음을 주목합시다. 기본적으로 return_sequences가 False인 경우에는 SimpleRNN은 마지막 시점의 은닉 상태만 출력합니다. 이번에는 return_sequences를 True로 지정하여 모든 시점의 은닉 상태를 출력해봅시다.

```python
rnn = SimpleRNN(3, return_sequences=True)
hidden_states = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
```

```tex
hidden states : [[[-0.99935263 -0.95985955 -0.9947519 ]
  [-0.9788508  -0.9995372  -0.9577954 ]
  [-0.03121292 -0.9998228  -0.9777464 ]
  [-0.9839102  -0.99654925  0.9253513 ]]], shape: (1, 4, 3)
```

(1, 4, 3)크기의 텐서가 출력됩니다. 앞서 입력 데이터는 (1, 4, 5)의 크기를 가지는 3D 텐서였고, 그 중 4가 시점(timesteps)에 해당하는 값이므로 모든 시점에 대해서 은닉 상태의 값을 출력하여 (1, 4, 3) 크기의 텐서를 출력하는 것입니다.

return_state가 True일 경우에는 return_sequences의 True/False 여부와 상관없이 마지막 시점의 은닉 상태를 출력합니다. 가령, return_sequences가 True이면서, return_state를 True로 할 경우 SimpleRNN은 두 개의 출력을 리턴합니다.

```python
rnn = SimpleRNN(3, return_sequences=True, return_state=True)
hidden_states, last_state = rnn(train_X)

print('hidden states : {}, shape: {}'.format(hidden_states, hidden_states.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
```

```tex
hidden states : [[[ 0.29839835 -0.99608386  0.2994854 ]
  [ 0.9160876   0.01154806  0.86181474]
  [-0.20252597 -0.9270214   0.9696659 ]
  [-0.5144398  -0.5037417   0.96605766]]], shape: (1, 4, 3)
last hidden state : [[-0.5144398  -0.5037417   0.96605766]], shape: (1, 3)
```

첫번째 출력은 return_sequences=True로 인한 출력으로 모든 시점의 은닉 상태입니다. 두번째 출력은 return_state=True로 인한 출력으로 마지막 시점의 은닉 상태입니다. 실제로 출력을 보면 모든 시점의 은닉 상태인 (1, 4, 3) 텐서의 마지막 벡터값이 return_state=True로 인해 출력된 벡터값과 일치하는 것을 볼 수 있습니다. (둘 다 [-0.5144398 -0.5037417 0.96605766])



# 3. LSTM 이해하기

실제로 SimpleRNN이 사용되는 경우는 거의 없습니다. 이보다는 LSTM이나 GRU을 주로 사용하는데, 이번에는 임의의 입력에 대해서 LSTM을 사용할 경우를 보겠습니다. 우선 return_sequences를 False로 두고, return_state가 True인 경우를 봅시다.

```python
lstm = LSTM(3, return_sequences=False, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))
```

```tex
hidden state : [[-0.00517232  0.11739909 -0.44045827]], shape: (1, 3)
last hidden state : [[-0.00517232  0.11739909 -0.44045827]], shape: (1, 3)
last cell state : [[-0.00652366  0.24878338 -0.8695111 ]], shape: (1, 3)
```

이번에는 SimpleRNN 때와는 달리, 세 개의 결과를 반환합니다. return_sequences가 False이므로 우선 첫번째 결과는 마지막 시점의 은닉 상태입니다. 그런데 LSTM이 SimpleRNN과 다른 점은 return_state를 True로 둔 경우에는 마지막 시점의 은닉 상태뿐만 아니라 셀 상태까지 반환한다는 점입니다. 이번에는 return_sequences를 True로 바꿔보겠습니다.

```python
lstm = LSTM(3, return_sequences=True, return_state=True)
hidden_state, last_state, last_cell_state = lstm(train_X)

print('hidden state : {}, shape: {}'.format(hidden_state, hidden_state.shape))
print('last hidden state : {}, shape: {}'.format(last_state, last_state.shape))
print('last cell state : {}, shape: {}'.format(last_cell_state, last_cell_state.shape))
```

```tex
hidden state : [[[ 0.00929032  0.1645535  -0.06458562]
  [ 0.02232158  0.12433957 -0.2470189 ]
  [ 0.0413315   0.1224088  -0.03404916]
  [ 0.00722066  0.26607004 -0.2263844 ]]], shape: (1, 4, 3)
last hidden state : [[ 0.00722066  0.26607004 -0.2263844 ]], shape: (1, 3)
last cell state : [[ 0.02635545  1.1380005  -0.37377644]], shape: (1, 3)
```

return_state가 True이므로 두번째 출력값이 마지막 은닉 상태, 세번째 출력값이 마지막 셀 상태인 것은 변함없지만 return_sequences가 True이므로 첫번째 출력값은 모든 시점의 은닉 상태가 출력됩니다.