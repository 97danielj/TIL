

[toc]

# 7) 문자 단위 RNN(Char RNN)

지금까지 배운 RNN은 전부 입력과 출력의 단위가 **단어 벡터**였습니다. 하지만 입출력의 단위를 단어 레벨(word-level)에서 문자 레벨(character-level)로 변경하여 RNN을 구현할 수 있습니다.

![img](https://wikidocs.net/images/page/48649/char_rnn1.PNG)

위의 그림은 문자 단위 RNN을 다 대 다(Many-to-Many) 구조로 구현한 경우, 다 대 일(Many-to-One) 구조로 구현한 경우 두 가지를 보여줍니다. 여기서는 이 두 가지 모두 구현해보겠습니다.

## **1. 문자 단위 RNN 언어 모델(Char RNNLM)**

이전 시점의 예측 문자를 다음 시점의 입력으로 사용하는 문자 단위 RNNㅇ언어 모델을 구현해봅시다. 앞서 배운 단어 단위 RNN언어 모델과 다른 점은 **단어 단위가 이닌 문자 단위를 입,출력으로 사용**하므로 임베딩층을 여기서는 사용하지 않겠습니다. 

데이터 : 이상한 나라의 앨리스 소설 (http://www.gutenberg.org/files/11/11-0.txt)



numpy의 np.random

