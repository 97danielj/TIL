[toc]

# os모듈, 파일과 디렉토리 활용

## 1. os 모듈의 다양한 함수

os 모듈은 내 컴퓨터의 디렉터리(폴더)나 경로, 파일 등을 활용하게 도와주는 모듈로 활용빈도가 굉장히 높다.
이 모듈이 제공하는 다양한 함수들에 대해 알아보자

### 1-1. os.getcwd() : 현재 작업 디렉토리 확인

```python
os.getcwd()
```

```tex
'C:\\Users\\JSWonner\\Desktop\\python_p\\stock_api'
```

### 1-2. os.chdir() : 현재 작업 디렉토리 변경

```python
os.chdir("D:/")
os.getcwd()
```

```tex
[Output]
'D:\\'
```

### 1-3. os.listdir() : 입력 경로 내의 모든 파일과 폴더명 리스트 반환

```
os.listdir("C:/Users/User/Desktop")
```

폴더는 폴더명, 파일은 확장자명까지 알려준다.

토글 메뉴

# [Python] os 모듈, 파일(file)과 디렉토리(directory)활용

업데이트: August 11, 2019

On This Page[1. os 모듈의 다양한 함수](https://yganalyst.github.io/data_handling/memo_1/#1-os-모듈의-다양한-함수)[1-1. os.getcwd() : 현재 작업 디렉토리 확인](https://yganalyst.github.io/data_handling/memo_1/#1-1-osgetcwd--현재-작업-디렉토리-확인)[1-2. os.chdir() : 현재 작업 디렉토리 변경](https://yganalyst.github.io/data_handling/memo_1/#1-2-oschdir--현재-작업-디렉토리-변경)[1-3. os.listdir() : 입력 경로 내의 모든 파일과 폴더명 리스트 반환](https://yganalyst.github.io/data_handling/memo_1/#1-3-oslistdir--입력-경로-내의-모든-파일과-폴더명-리스트-반환)[1-4. os.mkdir() : 폴더 생성](https://yganalyst.github.io/data_handling/memo_1/#1-4-osmkdir--폴더-생성)[1-5. os.makedirs() : 모든 하위 폴더 생성](https://yganalyst.github.io/data_handling/memo_1/#1-5-osmakedirs--모든-하위-폴더-생성)[1-6. os.remove() os.unlink() : 파일 삭제](https://yganalyst.github.io/data_handling/memo_1/#1-6-osremove-osunlink--파일-삭제)[1-7. os.rmdir() : 빈 폴더 삭제(가장 하위 폴더만)](https://yganalyst.github.io/data_handling/memo_1/#1-7-osrmdir--빈-폴더-삭제가장-하위-폴더만)[1-8. os.walk() : 경로, 폴더명, 파일명 모두 반환](https://yganalyst.github.io/data_handling/memo_1/#1-8-oswalk--경로-폴더명-파일명-모두-반환)[2. os.path 모듈의 다양한 함수](https://yganalyst.github.io/data_handling/memo_1/#2-ospath-모듈의-다양한-함수)[2-1. os.path.isdir() : 폴더 유무 판단](https://yganalyst.github.io/data_handling/memo_1/#2-1-ospathisdir--폴더-유무-판단)[2-2. os.path.isfile() : 파일 유무 판단](https://yganalyst.github.io/data_handling/memo_1/#2-2-ospathisfile--파일-유무-판단)[2-3. os.path.exists() : 파일이나 폴더의 존재여부 판단](https://yganalyst.github.io/data_handling/memo_1/#2-3-ospathexists--파일이나-폴더의-존재여부-판단)[2-4. os.path.getsize() : 파일의 크기(size) 반환](https://yganalyst.github.io/data_handling/memo_1/#2-4-ospathgetsize--파일의-크기size-반환)[2-5. os.path.split() os.path.splitext() : 경로와 파일 분리](https://yganalyst.github.io/data_handling/memo_1/#2-5-ospathsplit-ospathsplitext--경로와-파일-분리)[2-6. os.path.join() : 파일명과 경로를 합치기](https://yganalyst.github.io/data_handling/memo_1/#2-6-ospathjoin--파일명과-경로를-합치기)[2-7. os.path.dirname(), os.path.basename()](https://yganalyst.github.io/data_handling/memo_1/#2-7-ospathdirname-ospathbasename)[ref](https://yganalyst.github.io/data_handling/memo_1/#ref)

## 1. os 모듈의 다양한 함수

os 모듈은 내 컴퓨터의 디렉터리(폴더)나 경로, 파일 등을 활용하게 도와주는 모듈로 활용빈도가 굉장히 높다.
이 모듈이 제공하는 다양한 함수들에 대해 알아보자

### 1-1. `os.getcwd()` : 현재 작업 디렉토리 확인

```
os.getcwd()
[Output]
'C:\\Users\\User\\Desktop\\'
```

### 1-2. `os.chdir()` : 현재 작업 디렉토리 변경

```
os.chdir("D:/")
os.getcwd()
[Otuput]
'D:\\'
```

### 1-3. `os.listdir()` : 입력 경로 내의 모든 파일과 폴더명 리스트 반환

```
os.listdir("C:/Users/User/Desktop")
[Output]
['python_practice.py',
 '연구노트.hwp'
 '개인자료',
 '새 폴더',
 '공유자료',
 '데이터설명서모음',
 '크기비교 수정']
```

폴더는 폴더명, 파일은 확장자명까지 알려준다.

### 1-4. os.mkdir() : 폴더 생성

```
os.mkdir("C:/Users/User/Desktop/test")
os.listdir("C:/Users/User/Desktop/")
```

입력 경로의 마지막의 디렉토리 명으로 폴더를 생성한다.
이미 있는 파일명일 경우, 에러가 발생한다.

토글 메뉴

# [Python] os 모듈, 파일(file)과 디렉토리(directory)활용

업데이트: August 11, 2019

On This Page[1. os 모듈의 다양한 함수](https://yganalyst.github.io/data_handling/memo_1/#1-os-모듈의-다양한-함수)[1-1. os.getcwd() : 현재 작업 디렉토리 확인](https://yganalyst.github.io/data_handling/memo_1/#1-1-osgetcwd--현재-작업-디렉토리-확인)[1-2. os.chdir() : 현재 작업 디렉토리 변경](https://yganalyst.github.io/data_handling/memo_1/#1-2-oschdir--현재-작업-디렉토리-변경)[1-3. os.listdir() : 입력 경로 내의 모든 파일과 폴더명 리스트 반환](https://yganalyst.github.io/data_handling/memo_1/#1-3-oslistdir--입력-경로-내의-모든-파일과-폴더명-리스트-반환)[1-4. os.mkdir() : 폴더 생성](https://yganalyst.github.io/data_handling/memo_1/#1-4-osmkdir--폴더-생성)[1-5. os.makedirs() : 모든 하위 폴더 생성](https://yganalyst.github.io/data_handling/memo_1/#1-5-osmakedirs--모든-하위-폴더-생성)[1-6. os.remove() os.unlink() : 파일 삭제](https://yganalyst.github.io/data_handling/memo_1/#1-6-osremove-osunlink--파일-삭제)[1-7. os.rmdir() : 빈 폴더 삭제(가장 하위 폴더만)](https://yganalyst.github.io/data_handling/memo_1/#1-7-osrmdir--빈-폴더-삭제가장-하위-폴더만)[1-8. os.walk() : 경로, 폴더명, 파일명 모두 반환](https://yganalyst.github.io/data_handling/memo_1/#1-8-oswalk--경로-폴더명-파일명-모두-반환)[2. os.path 모듈의 다양한 함수](https://yganalyst.github.io/data_handling/memo_1/#2-ospath-모듈의-다양한-함수)[2-1. os.path.isdir() : 폴더 유무 판단](https://yganalyst.github.io/data_handling/memo_1/#2-1-ospathisdir--폴더-유무-판단)[2-2. os.path.isfile() : 파일 유무 판단](https://yganalyst.github.io/data_handling/memo_1/#2-2-ospathisfile--파일-유무-판단)[2-3. os.path.exists() : 파일이나 폴더의 존재여부 판단](https://yganalyst.github.io/data_handling/memo_1/#2-3-ospathexists--파일이나-폴더의-존재여부-판단)[2-4. os.path.getsize() : 파일의 크기(size) 반환](https://yganalyst.github.io/data_handling/memo_1/#2-4-ospathgetsize--파일의-크기size-반환)[2-5. os.path.split() os.path.splitext() : 경로와 파일 분리](https://yganalyst.github.io/data_handling/memo_1/#2-5-ospathsplit-ospathsplitext--경로와-파일-분리)[2-6. os.path.join() : 파일명과 경로를 합치기](https://yganalyst.github.io/data_handling/memo_1/#2-6-ospathjoin--파일명과-경로를-합치기)[2-7. os.path.dirname(), os.path.basename()](https://yganalyst.github.io/data_handling/memo_1/#2-7-ospathdirname-ospathbasename)[ref](https://yganalyst.github.io/data_handling/memo_1/#ref)

## 1. os 모듈의 다양한 함수

os 모듈은 내 컴퓨터의 디렉터리(폴더)나 경로, 파일 등을 활용하게 도와주는 모듈로 활용빈도가 굉장히 높다.
이 모듈이 제공하는 다양한 함수들에 대해 알아보자

### 1-1. `os.getcwd()` : 현재 작업 디렉토리 확인

```
os.getcwd()
[Output]
'C:\\Users\\User\\Desktop\\'
```

### 1-2. os.chdir() : 현재 작업 디렉토리 변경

```
os.chdir("D:/")
os.getcwd()
[Otuput]
'D:\\'
```

### 1-3. os.listdir() : 입력 경로 내의 모든 파일과 폴더명 리스트 반환

```
os.listdir("C:/Users/User/Desktop")
[Output]
['python_practice.py',
 '연구노트.hwp'
 '개인자료',
 '새 폴더',
 '공유자료',
 '데이터설명서모음',
 '크기비교 수정']
```

폴더는 폴더명, 파일은 확장자명까지 알려준다.

### 1-4. os.mkdir() : 폴더 생성

```
os.mkdir("C:/Users/User/Desktop/test")
os.listdir("C:/Users/User/Desktop/")
[Output]
['python_practice.py',
 '연구노트.hwp'
 '개인자료',
 '새 폴더',
 '공유자료',
 '데이터설명서모음',
 '크기비교 수정',
 'test']
```

입력 경로의 마지막의 디렉토리 명으로 폴더를 생성한다.
이미 있는 파일명일 경우, 에러가 발생한다.

```
os.mkdir("C:/Users/User/Desktop/test")
[Output]
---------------------------------------------------------------------------
FileExistsError                           Traceback (most recent call last)
<ipython-input-29-703c0a2ae4a0> in <module>
----> 1 os.mkdir("C:/Users/User/Desktop/tes1t")
FileExistsError: [WinError 183] 파일이 이미 있으므로 만들 수 없습니다: 'C:/Users/User/Desktop/tes1t'
```

### 1-5. os.makedirs() : 모든 하위 폴더 생성

경로의 제일 마지막에 적힌 폴더 하나만 생성하는 `os.mkdir()`과 달리 `os.makedirs()`함수는 경로의 모든폴더를 만들어 준다.

### 1-6. os.remove(), os.unlink() : 파일 삭제

```python
print(os.listdir("C:/Users/User/Desktop/tes1t/a/b"))
os.remove("C:/Users/User/Desktop/tes1t/a/b/test.txt")
print(os.listdir("C:/Users/User/Desktop/tes1t/a/b"))
```



### 1-7. os.rmdir() : 빈 폴더 삭제(가장 하위 폴더만)

빈 폴더만을 삭제해주며, 비어있지 않을 경우 에러 발생



### 1-8. os.walk() : 경로, 폴더명, 파일명 모두 반환

토글 메뉴

# [Python] os 모듈, 파일(file)과 디렉토리(directory)활용

업데이트: August 11, 2019

On This Page[1. os 모듈의 다양한 함수](https://yganalyst.github.io/data_handling/memo_1/#1-os-모듈의-다양한-함수)[1-1. os.getcwd() : 현재 작업 디렉토리 확인](https://yganalyst.github.io/data_handling/memo_1/#1-1-osgetcwd--현재-작업-디렉토리-확인)[1-2. os.chdir() : 현재 작업 디렉토리 변경](https://yganalyst.github.io/data_handling/memo_1/#1-2-oschdir--현재-작업-디렉토리-변경)[1-3. os.listdir() : 입력 경로 내의 모든 파일과 폴더명 리스트 반환](https://yganalyst.github.io/data_handling/memo_1/#1-3-oslistdir--입력-경로-내의-모든-파일과-폴더명-리스트-반환)[1-4. os.mkdir() : 폴더 생성](https://yganalyst.github.io/data_handling/memo_1/#1-4-osmkdir--폴더-생성)[1-5. os.makedirs() : 모든 하위 폴더 생성](https://yganalyst.github.io/data_handling/memo_1/#1-5-osmakedirs--모든-하위-폴더-생성)[1-6. os.remove() os.unlink() : 파일 삭제](https://yganalyst.github.io/data_handling/memo_1/#1-6-osremove-osunlink--파일-삭제)[1-7. os.rmdir() : 빈 폴더 삭제(가장 하위 폴더만)](https://yganalyst.github.io/data_handling/memo_1/#1-7-osrmdir--빈-폴더-삭제가장-하위-폴더만)[1-8. os.walk() : 경로, 폴더명, 파일명 모두 반환](https://yganalyst.github.io/data_handling/memo_1/#1-8-oswalk--경로-폴더명-파일명-모두-반환)[2. os.path 모듈의 다양한 함수](https://yganalyst.github.io/data_handling/memo_1/#2-ospath-모듈의-다양한-함수)[2-1. os.path.isdir() : 폴더 유무 판단](https://yganalyst.github.io/data_handling/memo_1/#2-1-ospathisdir--폴더-유무-판단)[2-2. os.path.isfile() : 파일 유무 판단](https://yganalyst.github.io/data_handling/memo_1/#2-2-ospathisfile--파일-유무-판단)[2-3. os.path.exists() : 파일이나 폴더의 존재여부 판단](https://yganalyst.github.io/data_handling/memo_1/#2-3-ospathexists--파일이나-폴더의-존재여부-판단)[2-4. os.path.getsize() : 파일의 크기(size) 반환](https://yganalyst.github.io/data_handling/memo_1/#2-4-ospathgetsize--파일의-크기size-반환)[2-5. os.path.split() os.path.splitext() : 경로와 파일 분리](https://yganalyst.github.io/data_handling/memo_1/#2-5-ospathsplit-ospathsplitext--경로와-파일-분리)[2-6. os.path.join() : 파일명과 경로를 합치기](https://yganalyst.github.io/data_handling/memo_1/#2-6-ospathjoin--파일명과-경로를-합치기)[2-7. os.path.dirname(), os.path.basename()](https://yganalyst.github.io/data_handling/memo_1/#2-7-ospathdirname-ospathbasename)[ref](https://yganalyst.github.io/data_handling/memo_1/#ref)

## 1. os 모듈의 다양한 함수

os 모듈은 내 컴퓨터의 디렉터리(폴더)나 경로, 파일 등을 활용하게 도와주는 모듈로 활용빈도가 굉장히 높다.
이 모듈이 제공하는 다양한 함수들에 대해 알아보자

### 1-1. `os.getcwd()` : 현재 작업 디렉토리 확인

```
os.getcwd()
[Output]
'C:\\Users\\User\\Desktop\\'
```

### 1-2. `os.chdir()` : 현재 작업 디렉토리 변경

```
os.chdir("D:/")
os.getcwd()
[Otuput]
'D:\\'
```

### 1-3. `os.listdir()` : 입력 경로 내의 모든 파일과 폴더명 리스트 반환

```
os.listdir("C:/Users/User/Desktop")
[Output]
['python_practice.py',
 '연구노트.hwp'
 '개인자료',
 '새 폴더',
 '공유자료',
 '데이터설명서모음',
 '크기비교 수정']
```

폴더는 폴더명, 파일은 확장자명까지 알려준다.

### 1-4. `os.mkdir()` : 폴더 생성

```
os.mkdir("C:/Users/User/Desktop/test")
os.listdir("C:/Users/User/Desktop/")
[Output]
['python_practice.py',
 '연구노트.hwp'
 '개인자료',
 '새 폴더',
 '공유자료',
 '데이터설명서모음',
 '크기비교 수정',
 'test']
```



입력 경로의 마지막의 디렉토리 명으로 폴더를 생성한다.
이미 있는 파일명일 경우, 에러가 발생한다.

```
os.mkdir("C:/Users/User/Desktop/test")
[Output]
---------------------------------------------------------------------------
FileExistsError                           Traceback (most recent call last)
<ipython-input-29-703c0a2ae4a0> in <module>
----> 1 os.mkdir("C:/Users/User/Desktop/tes1t")
FileExistsError: [WinError 183] 파일이 이미 있으므로 만들 수 없습니다: 'C:/Users/User/Desktop/tes1t'
```

### 1-5. `os.makedirs()` : 모든 하위 폴더 생성

경로의 제일 마지막에 적힌 폴더 하나만 생성하는 `os.mkdir()`과 달리 `os.makedirs()`함수는 경로의 모든폴더를 만들어 준다.

```
os.makedirs("C:/Users/User/Desktop/test/a/b")
```

실제로 확인해보면, `C:/Users/User/Desktop/test/a/b`이 생겨있다.

### 1-6. `os.remove()` `os.unlink()` : 파일 삭제

```
print(os.listdir("C:/Users/User/Desktop/tes1t/a/b"))
os.remove("C:/Users/User/Desktop/tes1t/a/b/test.txt")
print(os.listdir("C:/Users/User/Desktop/tes1t/a/b"))
[Output]
['test.txt']
[]
```

`os.unlink()`함수도 똑같이 동작한다.

### 1-7. `os.rmdir()` : 빈 폴더 삭제(가장 하위 폴더만)

빈 폴더만을 삭제해주며, 비어있지 않을 경우 에러 발생

```
os.rmdir("C:/Users/User/Desktop/test/a/b")
[Output]
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
<ipython-input-41-d718dffe3d52> in <module>
----> 1 os.rmdir("C:/Users/User/Desktop/test/a/b")

OSError: [WinError 145] 디렉터리가 비어 있지 않습니다: 'C:/Users/User/Desktop/test/a/b'
```

### 1-8. os.walk() : 경로, 폴더명, 파일명 모두 반환

`os.walk()`함수는 입력한 경로부터 그 경로 내의 모든 하위 디렉토리까지 하나하나 찾아다니며, 각각의 경로와 폴더명, 파일명들을 반환해 주는 함수이다.
generator로 반환해 주기 떄문에 for문이나 반복가능한(iterable) 함수 읽어야 한다.

```python
for path, direct, files in os.walk("c:/Users/User/Desktop"):
    print(path)
    print(direct)
    print(files)
```

```tex
[Output]
c:/Users/User/Desktop
['code_study', 'test']
['test1.txt', 'test2.txt', 'test3.txt', 'testtest.csv', '이슈 메모.hwp']
c:/Users/User/Desktop\code_study
['.ipynb_checkpoints', 'practice', 'review', '가져온자료']
['chunck.R', 'python_code_url.hwp']
```



## 2. os.path.join()

경로(패스)명 조작에 관한 처리를 모아둔 모듈로써 구현되어 있는 함수의 하나이다. 인수에 전달된 2개의 문자열을 결합하여, 1개의 경로로 할 수 있다.

**실제 사용법**

```python
#! /usr/bin/env python
import os

print("join(): " + os.path.join("/A/B/C", "file.py"))
```

```tex
join(): /A/B/C/file.py
```

### 1. join 활용법

**리스트를 이용한 경로 생성**
join()의 인수로 리스트를 전달하는 것도 가능하다.
다만, 주의점은 리스트를 전개해서 넘겨야한다는 것이다.
샘플 코드에서는 join()의 인수로써 전달된 list_path의 앞에 *(애스터리스크)를 붙여 리스트를 전개하고 있다.

```python
import os
list_path = ['C:\\', 'Users', 'user']
folder_path = os.path.join(*list_path) 
folder_path
```

```tex	
 'C:\\Users\\user'
```

### 2. 실행 파일의 어떤 한 디렉토리에 새로운 파일을 생성

 먼저 처음에 살펴 볼 것은 join()를 호출하고 있는 실행 파일의 어떤 한 디렉토리에 새로운 파일을 생성하는 방법이다. 실행 파일의 디렉토리를 취득하기 위해서는 아래와 같은 작성한다.

```python
print("join(): " + os.path.join(os.path.abspath(os.path.dirname(__file__)), "file.py"))
```

1. "__file__"에서 실행중의 파일을 표시
2. "os.path.dirnam"으로 실행 파일의 상대경로를 표시
3. "os.path.abspat"으로 위의 경로를 절대경로로 변환
 즉 의 코드를 실행시키면, 절대 경로를 얻을 수 있다.
    실행 결과는, 아래와 같다.

```tex
join(): /Users/XXX/Desktop/os-path-join/file.py
```

현재의 디렉토리에 새로운 파일을 생성
 이번에는 실행 파일이아닌 현재 디렉토리를 참고해보자. 스키립트를 실행한 시점에서는 실행 파일의 경로가 현재 디렉토리가 되지만, 디렉토리 이동등을 하려는 경우에는 현재의 디렉토리를 얻어야할 필요가 있다.
 현재 디렉토리를 얻어내는 방법은 다음과 같다.

```
#! /usr/bin/env python
import os

print("join(): " + os.path.join(os.getcwd(), "file.py"))
```

 **위와 같이 os.getcwd()에서 현재의 디렉토리를 취득할 수 있다.** 실행 결과는 "현재의 디렉토리(절대 경로)/file.py"이 된다.
※ 절대경로란, 제일 상위의 디렉토리로부터의 경로를 모두 적혀있는 경로이다.

### 3. 패스에 디렉토리의 구분 문자가 포함되어 있는 경우

결합하는 경로명에 디렉토리의 구분 문자("/"등)가 포함되어 있는 경우를 살펴보자.

```python
#! /usr/bin/env python

import os

print(os.path.join("dirA", "dirB", "/dirC"))

print(os.path.join("/dirA", "/dirB", "dirC"))

print(os.path.join("/dirA", "dirB", "dirC"))
```
세 가지 전부, /dirA/dirB/dirC로 결합될 것 같지만, 실행 결과를 보면 아래와 같이 출력된다.

```tex
/dirC
/dirB/dirC
/dirA/dirB/dirC
```

실제로는 join()은 디렉터리 구분 문자가 들어있으면, 그것을 root로 보는 성질이 있다. 대처법으로는 먼저 리스트를 이용해 문자열을 바꿔 쓰는 방법이 있다.

```python
#! /usr/bin/env python

import os

path = ["/dirA", "/dirB", "/dirC"]

path = [dir.replace("/", "") for dir in path] #리스트의 문자열 변경
print("join(): " + os.path.join(*path))
```

```tex
join(): dirA\dirB\dirC
```

