[toc]

# subplot, subplots 차이점

먼저, 하나의 전체 배열을 그린다고 생각하면 된다.

fig라는 **전체 배열(도화지)**에다가 ax를 사용해서 몇 행, 몇 열의 그래프를 그릴까?

**1행 2열**이면 **한 줄에 두 개** 그래프를 가지고 있다.

여기서 행과 열을 그릴 때, 우리는 좌표를 한꺼번(subplots)에 표현할지 아니면 좌표를 하나씩 (subplot) 표현할지 정하게 된다.

s의 존재 의미가 여러개인지 한 개인지를 나타낸다.

**subplots방법**

```python
#subplots의 2행 2열=> 4개의 subplot
x = np.arange(1,10)
fig, ax = plt.subplots(nrow=2,ncols=2) # 여러 개 그래프를 한번에 가능하다.
ax[0][0].plot(x) #1행 1열의 위치
ax[1][1].plot(x) # 2행 2열위치
plt.show()
```

**subplot방법**

```python
x = np.arange(1,10)
fig = plt.figure()
ax1 = plt.subplot(2,1,1) #2행 1열 첫 번째 그래프를 의미
ax2 = plt.subplot(2,1,2) #2행 1열 두 번째 그래프를 의미

ax1.plot(x)
ax2.plot(x,c='r')
plt.show()
```

