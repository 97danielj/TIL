[toc]

# datetime 모듈

파이썬은 내장 모듈인 datetime을 통해서 날짜와 시간 데이터를 처리를 지원하고 있습니다.

## 1. timedelta

datetime 내장모듈의 timedelta클래스는 기간을 표현하기 위해서 사용됩니다.

timedelta클래스의 생성자는 주,일,시분,초,밀리초, 마이크로 초를 인자로 받습니다.

```python
>>> from datetime import timedelta
>>> timedelta(days=5, hours=17, minutes=30)
datetime.timedelta(days=5, seconds=63000)
#timedelta객체는 내부적으로 일,초,마이크로 초 단위만 저장하기 때문에 위와 같은 해당 정보만 표시됩니다.

```

파이썬의 날짜/시간 계산은 다른언어에 비해서 매우 간결하고 직관적인데요. 바로 이 timedelta객체와 함께 산술/대소 연산자를 사용할 수 있기 때문입니다.

```python
>>> from datetime import date, timedelta
>>> today = date.today()
>>> today #현재 날짜
datetime.date(2020, 7, 18)
>>> one_week = timedelta(weeks=1)
>>> one_week
datetime.timedelta(days=7)
>>> next_week = today + one_week
>>> next_week
datetime.date(2020, 7, 25)
>>> two_weeks = one_week * 2
>>> two_weeks
datetime.timedelta(days=14)
>>> one_week < two_weeks
True
>>> two_weeks == timedelta(weeks=14)
True
>>> last_week = next_week - two_weeks
>>> last_week
datetime.date(2020, 7, 11)
```

`date`나 `time`, `datetime` 객체를 대상으로 유연한 날짜/시간 계산을 할 수 있습니다.



## 2. timezone

`datetime` 내장모듈의 `timezone`클래스는 시간대를 표현하기 위해서 사용됩니다. `timezone` 클래스의 생성자는 UTC 기준으로 시차를 표현하는 `timedelta` 객체를 인자로 받아 `timezone` 객체를 생성해줍니다. 예를 들어, 한국은 UTC 기준으로 9시간이 빠르므로 다음과 같이 `timezone` 객체를 생성할 수 있습니다.

```python
>>> from datetime import timedelta, timezone
#timezone : 시간대를 표현
>>> timezone(timedelta(hours=9))
datetime.timezone(datetime.timedelta(seconds=32400))

>>> timezone.utc
datetime.timezone.utc
```

`timezone` 객체는 앞으로 다룰 `time`과 `datetime` 객체의 `tzinfo` 속성값으로 사용됩니다.



## 3. date

`datetime` 내장 모듈의 `date`클래스는 날짜를 표현하는데 사용됩니다.

`date` 클래스의 생성자는 연, 월, 일 데이터를 인자로 받습니다.

```python
>>> from datetime import date
>>> date(2019, 12, 25)
datetime.date(2019, 12, 25) #날짜 정보를 나타낸다.

#오늘의 날짜를 얻기
>>> today = date.today()
datetime.date(2020, 7, 18)
#date객체를 YYYY-MM-DD형태의 문자열로 변환
>>> today.isoformat()

#YYYY-MM-DD 문자열을 date객체로 변환
>>> date.fromisoformat('2020-07-18')
datetime.date(2020, 7, 18)

>>> today = date.today()
>>> today.year
2020
>>> today.month
7
>>> today.day
18
```