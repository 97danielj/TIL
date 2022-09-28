[toc]

# 언어모델 BERT


> BERT : Pre-training of Deep Bidirectional Trnasformers for Language Understanding

구글에서 개발한 NLP(자연어처리) 사전 훈련 기술이며, 특정 분야에 국한된 기술이 아닌 **모든 자연어 처리 분야에서  좋은 성능을 내는 범용 Language Model**입니다.

 11개 이상의 자연어처리 과제에서 BERT가 최첨단 성능을 발휘한다고 하지만 그 이유는 잘 알려져 있지 않다고 합니다. 하지만 BERT는 지금까지 자연어처리에 활용하였던 앙상블 모델보다 더 좋은 성능을 내고 있어서 많은 관심을 받고 있는 언어모델 입니다.

## 1. 그래서 BERT가 과연 무엇인가.

처음에 BERT라는 모델을 접하였을 때, 단지 LSTM, CNN, 앙상블 모델로 개체명 인식, 텍스트 분류등의 과제를 시행하는 것과 같은 모델인 줄 알았지만 BERT는 **'사전 훈련 언어모델'** 입니다. 특정 과제를 수행하기 위한 모델의 성능은, 데이터가 충분히 많다면 Embedding이 큰 영향을 미칩니다. 단어의 의미를 잘 표현하는 벡터로 표현하는 Embedding된 단어들이 훈련과정에서 당연히 좋은 성능을 내겠죠?
이 임베딩 과정에서 BERT를 사용하는 것이고, BERT는 특정 과제를 하기 전 사전 훈련 Embedding을 통해 특정 과제의 성능을 더 좋게 할 수 있는 언어모델이라고 저는 이해하였습니다!
BERT등장 이전에는 데이터의 전처리 임베딩을 Word2Vec, GloVe, Fasttext 방식을 많이 사용했지만, 요즘의 고성능을 내는 대부분의 모델에서 BERT를 많이 사용하고 있다고 합니다.

예를 들어, 텍스트 분류모델을 만든다고 가정해 보겠습니다.

- BERT를 사용하지 않은 일반 모델과정은,
  - : 분류를 원하는 데이터 -> LSTM , CNN등의 머신러닝 모델 -> 분류
- BERT를 사용한 모델링 과정:
  - 관련 대량 코퍼스 -> BERT -> 분류를 원하는 데이터 -> LSTM, CNN 등의 머신러닝 모델 -> 분류
  - 대량의 코퍼스를 Encoder가 임베딩하고(언어 모델링), 이를 전이하여 파인튜닝하고 Task를 수행합니다.(NLP Task)

![img](https://blog.kakaocdn.net/dn/cEoPYe/btqBW0v9pJo/xM7PQl9BL0XAKX9fYuphw1/img.png)

대량 코퍼스로 BERT 언어모델을 적용하고, BERT언어모델 출력에 추가적인 모델(RNN, CNN 등의 머신러닝 모델)을 쌓아 원하는 Task를 수행하는 것 입니다. 이 때, 추가적인 모델을 복잡한 CNN, LSTM, Attention을 쌓지 않고 간단한 DNN모델만 쌓아도 Task 성능이 잘 나온다고 알려져 있고, DNN을 이용하였을 때와 CNN등 과 같은 복잡한 모델을 이용하였을 때의 성능 차이가 거의 없다고 알려져 있습니다.

## 2. BERT의 내부 동작 과정을 알아보자

### 1. Input

![img](https://blog.kakaocdn.net/dn/WFCfe/btqBWZ40Gmc/6FkuwsAGN9e7Uudmi03k4k/img.png)

> BERT Input : Token Embedding + Segment Embedding + Position Embedding

1. Token Embedding

: WordPiece 임베딩 방식 사용, 각 Char(문자)단위로 임베딩을 하고, 자주 등장하면서 가장 긴 길이의 sub-word를 하나의 단위로 만듭니다. 자주 등장하지 않는 단어는 다시 sub-word로 만듭니다. 이는 이전에 자주 등장하지 않았던 단어를 모조리 'OOV'처리하여 모델링의 성능을 저하했던 'OOV'문제도 해결 할 수 있습니다.

2. Segment Embedding

: Sentence Embedding, 토큰 시킨 단어들을 다시 하나의 문장으로 만드는 작업입니다. BERT에서는 두개의 문장을 구분자([SEP])를 넣어 구분하고 그 두 문장을 하나의 Segment로 지정하여 입력합니다. BERT에서는 이 한 세그먼트를 512 sub-word 길이로 제한하는데, 한국어는 보통 20 sub-word가 한 문장을 이룬다고 하며 대부분의 문장은 60 sub-word가 넘지 않는다고 하니 BERT를 사용할 때, 하나의 세그먼트에 128로 제한하여도 충분히 학습이 가능하다고 합니다.

3. Position Embedding

: BERT의 저자는 이전에 Transformer 모델을 발표하였는데, Transformer란 CNN, RNN 과 같은 모델 대신 Self-Attention 이라는 모델을 사용하는 모델입니다. BERT는 Transformer의 인코더, 디코더 중 인코더만 사용합니다. 
Transformer(Self-Attention) 참고글
Self Attention은 입력의 위치를 고려하지 않고 입력 토큰의 위치 정보를 고려합니다. 그래서 Transformer모델에서는 Sinusoid 함수를 이용하여 Positional encoding을 사용하고 BERT는 이를 따서 Position Encoding을 사용한다고 하는데,,, 무슨 말인지 잘 모르겠습니다! 간단하게 이해하면 Position encoding은 Token 순대로 인코딩 하는 것을 뜻합니다.

BERT는 위 세가지 임베딩을 합치고 이에 Layer정규화와 Dropout을 적용하여 입력으로 사용합니다.
BERT는 이미 총3.3억 단어(BookCorpus + Wikipedia Data)의 거대한 코퍼스를 정제하고, 임베딩하여 학습시킨 모델입니다. 이를 스스로 라벨을 만들고 준지도학습을 수행하였다고 합니다.

### 2. Pre-Training

: 데이터들을 임베딩하여 훈련시킬 데이터를 모두 인코딩 하였으면, 사전훈련을 시킬 단계입니다. 기존의 방법들은 보통 문장을 왼쪽에서 오른쪽으로 학습하여 다음 단어를 예측하는 방식이거나, 예측할 단어의 좌우 문맥을 고려하여 예측하는 방식을 사용합니다.
하지만 BERT는 언어의 특성을 잘 학습하도록,

MLM(Masked Language Model)
NSP(Next Sentence Prediction)

위 두가지 방식을 사용합니다. 

![img](https://blog.kakaocdn.net/dn/bgn7er/btqBVeIUJP5/RoaslDLh6TRSkk6zK5nqVK/img.png)

- MLM(Masked Language Model)

![img](https://blog.kakaocdn.net/dn/dTnfQQ/btqBVLfnFBV/rK5PCPsz2xX9t7qLEKUiF1/img.png)

: 입력 문장에서 임의로 토큰을 버리고(Mask), 그 토큰을 맞추는 방식으로 학습을 진행합니다.

- NSP(Next Sentence Prediction)

![img](https://blog.kakaocdn.net/dn/beTrc5/btqBTL8u19d/T1020drYaYApQP6TuKPjaK/img.png)

: 두 문장이 주어졌을 때, 두 문장의 순서를 예측하는 방식입니다. 두 문장 간 관련이 고려되야 하는 NLI와 QA의 파인 튜닝을 위해 두 문장의 연관을 맞추는 학습을 진행합니다.

### 3. Transfer Learning

: 학습된 언어모델을 전이학습시켜 실제 NLP Task를 수행하는 과정입니다. 실질적으로 성능이 관찰되는 부분이기도 합니다. BERT등장 이전에는 내가 NER문제를 풀고 싶다 하면 이에 관한 알고리즘이나 언어모델을 만들고, QA문제를 풀고 싶다 하면 이에 관한 알고리즘이나 언어모델을 따로 만들어야 했습니다. 하지만 BERT의 언어모델을 사용하여 전이학습시켜 원하는 Task를 수행해서 성능이 더 좋다는 것이 입증되었습니다! 여러모로 BERT는 대단한 모델이긴 한 것 같습니다. 기존에 언어모델을 만드는 부분은 스스로 라벨링을 하는 준지도 학습이었지만, 전이학습 부분은 라벨이 주어지는 지도학습 부분입니다. 전이학습(Transfer Learnin)은 BERT의 언어모델에 NLP Task를 위한 추가적인 모델을 쌓는 부분입니다. 이에 관해서는 다음장에 포스팅 하도록 하겠습니다.

![img](https://blog.kakaocdn.net/dn/EfY1i/btqBVeB8nT9/kVS2BX4Qc8kaFHWsr2E0SK/img.png)
