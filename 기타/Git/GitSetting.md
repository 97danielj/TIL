[toc]

# Git 기초

## Git 초기 설정

> 최초 한 번만 설정 합니다. 매번 Git을 사용할 때 마다 설정할 필요가 없습니다
>
1. 누가 커밋을 남겼는지 확인할 수 있도록 이름과 이메일을 설정합니다.

```bash
git config --global user.name "이름"
git config --global user,email "메일 주소"
```

2. 작성자가 올바르게 설정되었는지 확인 가능합니다.

```bash
git config --global -l
또는 
git config --global --list
```



## Git 기본 명령어

**1. git init**

```bash
$ git init
Initialized empty Git repository in C:/Users/kyle/git-practice/.git/

97daniel@daniel MINGW64 ~/git-practice (master)
```

* 현재 작업 중인 디겔터리를 Git으로 관리 한다는 명령어
* .git 이라는 숨김 폴더를 생성하고, 터미널에는 (master)라고 표기됩니다.
* 이미 Git저장소인 폴더 내에 또 다른 Git저장소를 만들지 않습니다.



**2. git status**

```bash
$ git status
On branch master

No commits yet

nothing to commit (create/copy files and use "git add" to track)
```

* Woriking Directory 와 Staiging Area에 있는 파일의 현재 상태를 알려주는 명령어
* 상태

  1. `Untracked` : Git이 관리하지 않는 파일 (한번도 Staging Area에 올라간 적 없는 파일)
  2. Tracked: Git이 관리하는 파일
     
     1. `Unmodified` : 최신 상태
      2. `Modified` : 수정되었지만 아직 Staging Area에는 반영하지 않은 상태
     3. `Staged` : Staging Area에 올라간 상태



**3. git add**

```git add a.txt```

- Working Directory에 있는 파일을 Staging Area로 올리는 명령어
- Git이 해당 파일을 추적(관리)할 수 있도록 만듭니다.
- `Untracked, Modified → Staged` 로 상태를 변경합니다.



**4. git commit**

```bash
$ git commit -m "first commit"
[master (root-commit) c02659f] first commit
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 a.txt
```

- Staging Area에 올라온 파일의 변경 사항을 하나의 버전(커밋)으로 저장하는 명령어
- `커밋 메세지`는 현재 변경 사항들을 잘 나타낼 수 있도록 `의미` 있게 작성하는 것을 권장합니다.
- 각각의 커밋은 `SHA-1` 알고리즘에 의해 반환 된 고유의 해시 값을 ID로 가집니다.
- `(root-commit)` 은 해당 커밋이 최초의 커밋 일 때만 표시됩니다. 이후 커밋부터는 사라집니다.



**5. git log**

- 커밋의 내역(`ID, 작성자, 시간, 메세지 등`)을 조회할 수 있는 명령어
- 옵션
  - `--oneline` : 한 줄로 축약해서 보여줍니다.
  - `--graph` : 브랜치와 머지 내역을 그래프로 보여줍니다.
  - `--all` : 현재 브랜치를 포함한 모든 브랜치의 내역을 보여줍니다.
  - `--reverse` : 커밋 내역의 순서를 반대로 보여줍니다. (최신이 가장 아래)
  - `-p` : 파일의 변경 내용도 같이 보여줍니다.
  - `-2` : 원하는 갯수 만큼의 내역을 보여줍니다. (2 말고 임의의 숫자 사용 가능)





