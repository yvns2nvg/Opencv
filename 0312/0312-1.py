import cv2
import numpy as np
import glob

# ---------------------------------------------------------
# [초기 설정] 체크보드 규격 및 알고리즘 정밀도 설정
# ---------------------------------------------------------
# 체크보드 내부 코너 개수 (검은색/흰색 사각형이 교차하는 점의 개수: 가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 물리적 크기 (25mm)
square_size = 25.0

# 찾아낸 코너의 좌표를 소수점 단위까지 더욱 정밀하게 다듬기 위한 조건 (Sub-pixel accuracy)
# 최대 30번 반복(MAX_ITER)하거나 오차가 0.001 이하(EPS)가 되면 탐색 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 세계(3차원 공간)에서의 체크보드 코너 좌표 생성 (Z축은 0으로 가정)
# 예: (0,0,0), (25,0,0), (50,0,0) ... 와 같이 25mm 간격의 격자 좌표를 미리 만들어둠
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 앞으로 처리할 여러 사진에서 찾은 좌표들을 차곡차곡 모아둘 빈 바구니(리스트) 두 개
objpoints = [] # 실제 3D 세계 좌표를 담을 리스트 (objp가 여러 개 들어감)
imgpoints = [] # 사진 상에서 찾은 2D 픽셀 좌표를 담을 리스트

# 보정용 이미지 파일들이 있는 폴더 경로 설정 (left01.jpg ~ left13.jpg 모두 찾기)
images = glob.glob("0312/images/calibration_images/left*.jpg")
img_size = None

# =========================================================
# 1. 체크보드 코너 검출 과정
# =========================================================
# 찾은 이미지 파일들을 하나씩 모두 꺼내서 반복 처리합니다.
for fname in images:
    img = cv2.imread(fname)                      # 컬러 이미지로 불러오기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 코너 검출을 위해 흑백 이미지로 변환
    
    # 이미지의 가로/세로 픽셀 해상도 사이즈 기억해두기 (나중에 캘리브레이션 함수에 필요함)
    if img_size is None:
        img_size = gray.shape[::-1]

    # [핵심] 컴퓨터 비전 알고리즘으로 체스보드의 코너들을 1차적으로 쓱 훑어서 찾아냅니다.
    # 성공하면 ret은 True, corners에는 찾은 픽셀 좌표들이 담깁니다.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 성공적으로 9x6개 모두 찾았다면?
    if ret == True:
        # 실제 3D 좌표 세트(미리 만들어둔 objp)를 정답 바구니에 담습니다.
        objpoints.append(objp)
        
        # 1차적으로 찾은 픽셀 좌표(corners)를 아까 설정한 criteria 조건에 맞춰서
        # 소수점 단위까지 아주 정밀하게 위치를 한 번 더 깎고 다듬습니다. (정확도 상승)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 다듬어진 최종 2D 픽셀 좌표를 바구니에 담습니다.
        imgpoints.append(corners2)

cv2.destroyAllWindows()

# =========================================================
# 2. 카메라 캘리브레이션 (파라미터 추출)
# =========================================================
# 위에서 열심히 모은 정답 3D 좌표(objpoints)와 이미지 2D 좌표(imgpoints)를 바탕으로
# 이 카메라가 어떤 왜곡을 가졌고, 렌즈 초점거리가 어떤지(내부 행렬) 역산해냅니다!
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 계산된 결과 출력
print("Camera Matrix K (카메라 내부 행렬: 초점거리 및 센서 중심점 정보):")
print(K)

print("\nDistortion Coefficients (렌즈 왜곡 계수: 방사형 및 접선형 렌즈 찌그러짐 정도):")
print(dist)

# =========================================================
# 3. 실제 이미지 왜곡 보정 시각화
# =========================================================
# 계산된 수치가 정말 맞는지 확인해보기 위해, 첫 번째 사진(test_img)을 꺼냅니다.
test_img = cv2.imread(images[0])

# [핵심] 아까 구한 카메라 행렬(K)과 왜곡 계수(dist)를 이용해 둥글게 휘어진 사진을 평평하게 폅니다.
undistorted_img = cv2.undistort(test_img, K, dist, None, K)

# 렌즈 왜곡이 펴지기 전/후 사진을 나란히 화면에 띄워 눈으로 비교합니다.
cv2.imshow('Original Image (Before)', test_img)
cv2.imshow('Undistorted Image (After)', undistorted_img)

print("\n보정 결과 시각화 중... 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()