import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (가로, 세로)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 3D 좌표 생성 (모든 이미지에서 동일한 격자 구조 가짐)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 실제 좌표(objpoints)와 이미지 좌표(imgpoints)를 저장할 리스트
objpoints = [] 
imgpoints = [] 

# 이미지 경로 설정 (left01.jpg ~ left13.jpg)
images = glob.glob("0312/images/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_size is None:
        img_size = gray.shape[::-1]

    # 이미지에서 체크보드 코너 검출
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출에 성공한 경우 좌표 구성
    if ret == True:
        objpoints.append(objp)
        
        # 코너 좌표 정밀화 (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 검출 결과 시각화 (확인용)
        # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imshow('Chessboard Corners', img)
        # cv2.waitKey(100)

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 내부 행렬 K와 왜곡 계수(dist) 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화 
# -----------------------------
# 테스트할 원본 이미지 로드 (예: left01.jpg)
test_img = cv2.imread(images[0])

# 왜곡 보정 적용 
undistorted_img = cv2.undistort(test_img, K, dist, None, K)

# 결과 비교 출력
cv2.imshow('Original Image', test_img)
cv2.imshow('Undistorted Image', undistorted_img)

print("\n보정 결과 시각화 중... 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()