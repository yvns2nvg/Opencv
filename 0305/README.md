# 0305 OpenCV 예제 모음


## 요구 환경
- Python 3.x
- 패키지: `opencv-python`, `numpy`

---

**목차**
- [0305/0305-1.py](0305/0305-1.py) — 원본과 그레이스케일 병합 출력
- [0305/0305-2.py](0305/0305-2.py) — 간단 페인팅(브러시) 앱
- [0305/0305-3.py](0305/0305-3.py) — ROI 선택 및 저장

---
---

## 0305-1.py

- 파일: [0305/0305-1.py](0305/0305-1.py)

- 전체 코드:

```python
import cv2 as cv
import numpy as np
import sys

# 이미지 읽기 및 예외 처리
source_img = cv.imread('0305/soccer.jpg')     # 로컬 이미지 파일 불러오기

if source_img is None:                   # 파일이 없거나 경로가 틀린 경우
    sys.exit('이미지를 불러오는 데 실패했습니다.')

# 이미지 처리 (색상 변환)
# 컬러(BGR) 이미지를 흑백(Gray) 영상으로 변환
gray_img = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)

# 흑백(1채널) 이미지를 결합을 위해 컬러 형식(3채널)으로 재변환
gray_to_bgr = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)

# 이미지 결합
# 두 이미지를 가로 방향(Horizontal)으로 나란히 연결
combined_view = np.hstack((source_img, gray_to_bgr))

# 화면 출력 및 종료 제어
cv.imshow('Original vs Grayscale', combined_view)

cv.waitKey(0)           # 키보드 입력이 있을 때까지 대기
cv.destroyAllWindows()  # 모든 윈도우 창 닫기 및 메모리 해제
```

- 핵심 기능 코드(요구사항):
  - `source_img = cv.imread('0305/soccer.jpg')` — 이미지 파일을 메모리로 읽어옵니다.
  - `gray_img = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)` — BGR 컬러 영상을 그레이스케일(단일 채널)로 변환합니다.
  - `gray_to_bgr = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)` — 그레이 영상을 3채널 BGR로 변환해 결합 시 채널 수를 맞춥니다.
  - `combined_view = np.hstack((source_img, gray_to_bgr))` — 원본과 그레이스케일(3채널)을 가로로 이어 붙입니다.
  - `cv.imshow('Original vs Grayscale', combined_view)` / `cv.waitKey(0)` — 결과 창에 표시하고 키 입력을 기다려 창을 닫습니다.

- 결과 이미지(예시):

<img width="2864" height="996" alt="result-1" src="https://github.com/user-attachments/assets/074510bf-7930-4ec9-a1ca-51c45e4dcaf0" />



---

## 0305-2.py

- 파일: [0305/0305-2.py](0305/0305-2.py)

- 전체 코드:

```python
import cv2 as cv
import sys

# 이미지 읽기 및 예외 처리
canvas = cv.imread('0305/soccer.jpg')

if canvas is None:
    sys.exit('이미지를 불러오지 못했습니다. 경로를 확인해주세요.')

# 브러시 속성 설정 (BGR 순서)
pointer_radius = 5      # 붓의 기본 반지름
color_blue = (255, 0, 0) 
color_red = (0, 0, 255)

# 마우스 콜백 함수 정의
is_drawing = False       # 마우스 버튼 눌림 상태 확인
prev_x, prev_y = -1, -1   # 직전 좌표 저장용

def on_mouse_paint(event, x, y, flags, param):
    global pointer_radius, is_drawing, prev_x, prev_y
    
    # 마우스를 눌렀을 때: 그리기 시작 및 시작점 저장
    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        is_drawing = True
        prev_x, prev_y = x, y

    # 마우스 이동 시: 이전 좌표와 현재 좌표를 선(line)으로 연결
    elif event == cv.EVENT_MOUSEMOVE:
        if is_drawing:
            # 좌클릭은 파랑, 우클릭은 빨강
            current_color = color_blue if (flags & cv.EVENT_FLAG_LBUTTON) else color_red
            
            cv.line(canvas, (prev_x, prev_y), (x, y), current_color, pointer_radius * 2)
            
            # 현재 좌표를 다음 선의 시작점으로 업데이트
            prev_x, prev_y = x, y

    # 마우스 버튼을 뗐을 때: 그리기 중단
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        is_drawing = False

    cv.imshow('Painting App', canvas)

# 윈도우 생성 및 마우스 이벤트 연결
cv.namedWindow('Painting App')
cv.setMouseCallback('Painting App', on_mouse_paint)

# 메인 루프 실행
while True:
    cv.imshow('Painting App', canvas)
    
    # 키보드 입력 대기
    user_input = cv.waitKey(1) & 0xFF 
    
    # 프로그램 종료
    if user_input == ord('q'): 
        break
        
    # 브러시 크기 증가 (최대 15)
    elif user_input in [ord('+'), ord('=')]: 
        pointer_radius = min(15, pointer_radius + 1) 
        
    # 브러시 크기 감소 (최소 1)
    elif user_input in [ord('-'), ord('_')]: 
        pointer_radius = max(1, pointer_radius - 1)  

# 자원 해제
cv.destroyAllWindows()
```

- 핵심 기능 코드(요구사항):
  - 초기 붓 크기: `pointer_radius = 5` — 프로그램 시작 시 기본 브러시 반지름을 설정합니다.
  - 크기 조절: `pointer_radius = min(15, pointer_radius + 1)` / `pointer_radius = max(1, pointer_radius - 1)` — `+`/`-` 키로 반지름을 1씩 증감하며 1~15 범위로 제한합니다.
  - 좌/우클릭 색상 분기: `current_color = color_blue if (flags & cv.EVENT_FLAG_LBUTTON) else color_red` — 좌클릭은 파란색, 우클릭은 빨간색으로 그립니다.
  - 연속 그리기: `cv.line(canvas, (prev_x, prev_y), (x, y), current_color, pointer_radius * 2)` — 드래그 시 이전 좌표와 현재 좌표를 연결해 부드러운 선을 만듭니다.
  - 종료: `if user_input == ord('q'): break` — `q` 키로 창을 닫고 프로그램을 종료합니다.

- 결과 이미지(예시):

<img width="1422" height="998" alt="result-2" src="https://github.com/user-attachments/assets/1d3bdb31-1594-4b56-ae31-b788e9087ab0" />



---

## 0305-3.py

- 파일: [0305/0305-3.py](0305/0305-3.py)

- 전체 코드:

```python
import cv2 as cv
import sys

# 이미지 로드 및 초기 설정
# 0305 폴더 내의 이미지를 불러와 작업 화면으로 설정
source_view = cv.imread('0305/girl_laughing.jpg')

if source_view is None:
    sys.exit('이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.')

# 상태 관리를 위한 변수들
clean_copy = source_view.copy()  # 드래그 시 잔상을 지우기 위한 원본 복사본
start_x, start_y = -1, -1        # 마우스 클릭 시작 지점
is_selecting = False             # 드래그 진행 상태 확인
selected_roi = None              # 최종적으로 잘라낸 영역 데이터

# 마우스 콜백 함수 정의
def on_mouse_roi(event, x, y, flags, param):
    global start_x, start_y, is_selecting, source_view, selected_roi
    
    # 마우스 왼쪽 버튼 클릭 시 시작점 저장
    if event == cv.EVENT_LBUTTONDOWN:
        is_selecting = True
        start_x, start_y = x, y
        
    # 마우스 드래그 중 실시간으로 초록색 가이드 사각형 시각화
    elif event == cv.EVENT_MOUSEMOVE:
        if is_selecting:
            # 이전 프레임의 사각형을 지우기 위해 원본을 매번 새로 복사
            source_view = clean_copy.copy()
            # 현재 마우스 위치까지 실시간 사각형 표시
            cv.rectangle(source_view, (start_x, start_y), (x, y), (0, 255, 0), 2)
        
    # 마우스 버튼을 뗐을 때 영역 확정 및 추출
    elif event == cv.EVENT_LBUTTONUP:
        is_selecting = False
        
        # 선택이 완료된 영역은 빨간색 사각형으로 강조
        cv.rectangle(source_view, (start_x, start_y), (x, y), (0, 0, 255), 2)
        
        # 슬라이싱을 이용해 영역 추출 (역방향 드래그도 가능하도록 min/max 사용)
        y_range = slice(min(start_y, y), max(start_y, y))
        x_range = slice(min(start_x, x), max(start_x, x))
        selected_roi = clean_copy[y_range, x_range]
        
        # 추출된 영역이 존재하면 새로운 창에 표시
        if selected_roi.size > 0:
            cv.imshow('Extracted ROI', selected_roi)

# 윈도우 생성 및 마우스 이벤트 연결
cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', on_mouse_roi)

print("[조작법] 드래그: 영역 선택 | R: 초기화 | S: 저장 | Q: 종료")

# 메인 실행 루프
while True:
    cv.imshow('Select ROI', source_view)
    
    user_key = cv.waitKey(1) & 0xFF
    
    # 'r' 키: 선택 영역 초기화 및 화면 리셋
    if user_key == ord('r'):
        source_view = clean_copy.copy()
        selected_roi = None
        if cv.getWindowProperty('Extracted ROI', 0) >= 0:
            cv.destroyWindow('Extracted ROI')
        print("화면이 초기화되었습니다.")

    # 's' 키: 선택된 영역을 이미지 파일로 저장
    elif user_key == ord('s'):
        if selected_roi is not None and selected_roi.size > 0:
            cv.imwrite('0305/captured_roi.jpg', selected_roi)
            print("선택 영역이 '0305/captured_roi.jpg'로 저장되었습니다.")
        else:
            print("저장할 영역이 없습니다. 먼저 마우스로 드래그하세요.")

    # 'q' 키: 프로그램 종료
    elif user_key == ord('q'):
        break

# 프로그램 종료 시 모든 창 닫기
cv.destroyAllWindows()
```

- 핵심 기능 코드(요구사항):
  - 이미지 로드: `source_view = cv.imread('0305/girl_laughing.jpg')` — 표시 및 ROI 선택을 위한 이미지를 읽어옵니다.
  - 마우스 콜백 등록: `cv.setMouseCallback('Select ROI', on_mouse_roi)` — 마우스 이벤트를 받아 사각형 선택을 처리합니다.
  - 실시간 가이드 사각형: `cv.rectangle(source_view, (start_x, start_y), (x, y), (0, 255, 0), 2)` — 드래그 중 녹색 테두리로 선택 영역을 시각화합니다.
  - ROI 추출: `selected_roi = clean_copy[y_range, x_range]` (슬라이싱으로 추출) — 선택한 좌표 범위를 잘라내어 별도 창에 표시하거나 저장할 수 있습니다.
  - 리셋: `if user_key == ord('r'):` → `source_view = clean_copy.copy()` — `r` 키로 선택을 초기화하고 원본 상태로 되돌립니다.
  - 저장: `cv.imwrite('0305/captured_roi.jpg', selected_roi)` — `s` 키로 선택한 ROI를 파일로 저장합니다.

- 결과 이미지(예시):
<img width="2166" height="1018" alt="result-3" src="https://github.com/user-attachments/assets/ba8434a0-dab0-4579-892d-3f715d29cc91" />

<img width="1482" height="980" alt="result-4" src="https://github.com/user-attachments/assets/08e65c8d-90b5-4966-9218-41866c294f48" />


---


