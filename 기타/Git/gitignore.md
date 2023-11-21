[toc]
# .gitignore

> 특정 파일 혹은 폴더에 대해 Git이 버전 관리를 못하도록 지정하는 것. 즉 추적을 무시

## (1) .gitignore에 작성하는 목록

* 민감한 개인 정보가 담긴 파일(전화번호, 계좌번호, 각종 비밀번호, API KEY 등)
* OS(운영체제)에서 활용되는 파일
* IDE(통합 개발 환경) 혹은 Text Editor 등에서 활용되는 파일
* 개발 언어 혹은 프레임워크에서 사용되는 파일
*  용량이 너무 커서 제외해야 되는 파일

## (2) .gitignore 파일 작성시 주의 사항

* 반드시 이름은 .gitignore로 지정. 앞의 점(.)은 숨김 파일이라는 뜻입니다.
* .gitignore 파일은 .git 폴더와 동일한 위치에 생성합니다.
* **제외 하고 싶은 파일 반드시 git add 전에 .gitignore에 작성합니다.**



## (3) .gitignore 파일 예제

작성하는 몇 가지 규칙

- 주석은 #로 표기
- 표준 Glob 패턴을 사용
- 슬래시(/)로 시작하면 하위 디렉터리에 적용되지(recursivity) 않음
- 디렉터리는 슬래시(/)를 끝에 사용하는 것으로 표현
- 느낌표(!)로 시작하는 패턴의 파일은 무시하지 않음

```bash
# : comments

# ignore all .a files
*.a

# exclude lib.class from "*.a", meaning al lib.a are still tracked
!lib.a

# only ignore The TODO file in the current directory, not subdir/TODO
/TODO

# ignore all json files whose name begin with 'temp-'
temp-*.json

# only ignore the build.log file in current directory, not those in its subdirectories
/build.log

# specify a folder with slash in the end
# ignore all files in the build/ directory

build/

# ignore all .pdf files in the doc/ directory
# /** matches 0 or more directories</span>

doc/**/*. pdf

```



## (4) .gitignore을 쉽게 사용

[gitignore.io]: https://www.toptal.com/developers/gitignore

라는 사이트에서 쉽게 .gitignore파일을 생성 가능하다.

