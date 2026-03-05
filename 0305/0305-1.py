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