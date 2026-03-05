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