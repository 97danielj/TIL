\#include <opencv2/opencv.hpp> : OpenCV API를 사용 위한 기본 헤더 파일 포함

| 데이터 형 |         설명         | depth 값 |
| :-------: | :------------------: | :------: |
|   CV_8U   | uchar(unsigned char) |    0     |
|   CV_8S   |     signed char      |    1     |
|  CV_16U   |  unsigned short int  |    2     |
|  CV_16S   |   signed short int   |    3     |
|  CV_32S   |         int          |    4     |
|  CV_32F   |        float         |    5     |
|  CV_64F   |        double        |    6     |

cv:: opencvd의 네임스페이스

1. Point 클래스

- 포인트 객체간 덧셈. 스칼라 곱셈, 비교연산 가능, 단 객체간 곱셈은 불가

- cv::Point_<int> pt1(100,200);

- cv::Point2i pt2(100,200);

- pt7.dot(pt8);

- 3차원 자료를 나타내기 위한 자료형

  - Point3_<int> pt(100,200,300)

  - Point3f pt2(0.3f,0.f,15.7f)

2.  Size클래스

- Size_<int> sz1(100,200)

- Size2f sz5(0.3f,0.f)

- sz1.area() : 사각형 객체 넓이 반환

3. Rect_클래스

- 2차원의 사각형 정보를 나타내기 위한 템플릿 클래스

  - x좌표, y좌표, 너비, 높이
    - Rect2f rect(10.f,20.f,30.f,40.f)

  - 시작좌표(pt1), 크키(sz)
  - 시작좌표(pt1), 종료좌표(pt2)
    - Rect2f rect2(pt1,pt2)
  - Rect2d rect6 = rect1 & (Rect)rect2

4. Vec 클래스 
   - 벡터 클래스
   - Vec2i(리터럴1,리터럴2,리터럴3)
   - Vec4f(리터럴1,리터럴2,리터럴3) : 굳이 개수 맞출 필용벗다.
   - v3.mul(v7) : 행렬 곱
5. Scalar_ 클래스
   - Vec클래스중 Vec<Tp,4>에서 파생된 템플릿 클래스
   - 특별히 화소의 값을 지정하기 위한 자료형의 정의
   - 파랑, 초록, 빨간,투명도의 4개의 값 저장
   - 값을 초기화 하지 않았으면 0. 즉 항상 4개의 원소를 가진다.
6. RotatedRect 클래스
7. Mat클래스

- 행렬을 다루기 위한 클래스
- Mat객체 생성자
  - Mat(int rows, int cols, int type)
    - Mat m1(2, 3, CV_8U)
  - Mat(int rows, int cols, int type, const Scalar)
    - Mat m2(3,4,CV_8U, Scalar(300))
    -  __saturate cast연산__
  - Mat(Size size, int type, const Scalar)
    - Mat m3(size(3,4),CV_32F,data) #4행,3열이다.
- Mat객체의 자료형 지정 오류 예
  - int data[] 를 short형 행렬에 넣으면 short형 2개를 int형 데이터 하나가 사용하여 1나씩 걸러 삽입된다.
  - saturate cast연산
  - Mat::ones(3,5, CV_8UC1) : 모든원소 1, 데이터 타입 1byte
  - Mat::eye(3,5, CV_8UC1))  : 지정된 크기와 타입의 단위 행렬을 반환한다.(단위행렬)
  - Mat::zeros(3,5, CV_8UC1)) : 행렬의 원소를 0으로 초기화

9. Mat__클래스를 이용한 초기화
   - Mat_<int>  m1(2,4)


10. Matx클래스를 이용한 초기화

    - Matx<int,3,3> m1(1,2,3,4,5,6,7,8,9); // Matx객체 선언과 동시에 원소 초기화

    - 접근 : m2(2,3)


11. Mat클래스의 다양한 속성
    - Mat::step :  행렬의 한 행이 차지하는 바이트 수. 채널이 여러개 있을 경우 그만큼 많아진다.
    - Mat::depth() : 행렬의 깊이(행렬의 자료형) 값 반환
    - Mat::elemSize() : 행렬의 한 원소에 대한 바이트 크기 반환
    - Mat::elemSize1() : 행렬의 한 원소의 한 채널에 대한 바이트 크기 반환

12. Mat클래스의 크기 및 형태 변경
    - Mat::resize(size_t,sz) 
    - Mat:reshape(chal,rows)
13. Mat복사 및 자료형 변환
    - Mat clone () : 행렬 데이터와 같은 값을 복사해서 새로운 행렬을 반환한다
    - void copyTo(mat) : 행렬 데이터를 인자로 입력된 mat행렬에 복사한다
14. Vector클래스
    1. 시퀀스 컨테이너
    2. 동적 배열구조, 원소의 추가 및 삭제가 용이
    3. vector(size_type _Count, value)
       1. 생성자
       2. COUNT: 원소 개수 
       3. Value : 초기값
    4. iterator insert()  : 원소를 삽입