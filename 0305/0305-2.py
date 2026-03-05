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