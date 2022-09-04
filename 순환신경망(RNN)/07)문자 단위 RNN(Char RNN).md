[toc]

# 7) 문자 단위 RNN(Char RNN)

지금까지 배운 RNN은 전부 입력과 출력의 단위가 단어 벡터였습니다. 하지만 입출력의 단위를 단어 레벨(word-level)에서 문자 레벨(character-level)로 변경하여 RNN을 구현할 수 있습니다.

![img](https://wikidocs.net/images/page/48649/char_rnn1.PNG)

위의 그림은 문자 단위 RNN을 다 대 다(Many-to-Many) 구조로 구현한 경우, 다 대 일(Many-to-One) 구조로 구현한 경우 두 가지를 보여줍니다. 여기서는 이 두 가지 모두 구현해보겠습니다. 첫번째로 구현할 것은 다 대 다 구조를 이용한 언어 모델입니다.