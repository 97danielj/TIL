# 기말고사

## 5. 기본 행렬 연산

### 5.1 기본배열 처리 함수

- 컬러 영상
  - BGR의 각기 독립적인 2차원 정보
  - 2차원정보 3개를 갖는 컬러영상을 표현

```c++
flip(image, x_axis,0);
repeat(image,1,2,rep_img); //가로 2번 반복
```

### 5.2 채널 처리 함수

```c++
void merge(bgr_arr,3,bg); //여러개의 단일 채널 배열로 다중 채널의 배열을 합성
void split(image,bgr); //다중 채널 배열을 여러 개의 단일채널 배열로 분리한다.
out = mixChannels()
```

