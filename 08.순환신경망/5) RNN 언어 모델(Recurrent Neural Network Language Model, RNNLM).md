[TOC]

# 5) RNN 언어 모델(Recurrent Neural Network Language Model, RNNLM)

<hr/>

RNN을 이용하여 언어 모델을 구현한 RNN 언어 모델에 대해서 배웁니다.

## **1. RNN 언어 모델(Recurrent Neural Network Language Model, RNNLM)**

앞서 **n-gram언어 모델과 NNLM은 고정된 개수의 단어만을 입력을 받아야한다는 단점**이 있엇습니다.하지만 **시점**이라는 개념을 도입한 **RNN**으로 언어 모델을 만들면 입력의 길이를 고정하지 않을 수 있습니다. 이처럼 RNN으로 만든 언어 모델을 RNNLM이라고 합니다.

> RNNLM은 시점의 개념을 도입하여 입력의 길이를 고정하지 않은 언어 모델이다.

RNNLM이 언어 모델링을 학습하는 과정을 보겠습니다. 이해를 위해 간소화 된 형태로 설명합니다.

- 예문 : 'what will the fat cat sit on'

예를 들어 훈련 코퍼스에 위와 같은 문장이 있다고 해봅시다. **언어모델은 주어진 단어 시퀀스로부터 다음단어를 예측하는 모델입니다.** 아래의 그림은 RNN이 어떻게 이전 시점의 단어들과 현재 시점의 단어로 다음 단어를 예측하는지를 보여줍니다.



