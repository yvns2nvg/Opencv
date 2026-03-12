# OpenCV Image Formation 실습 과제 (0312)

이 저장소는 `L02_Image_Formation` 강의 실습 과제 3가지의 코드가 포함되어 있습니다.

---

## 📌 과제 1: 체크보드 기반 카메라 캘리브레이션 (`0312-1.py`)

### 1. 문제 정의
*   여러 장의 체크보드 이미지에서 코너를 검출하고 실제 3D 좌표와의 대응 관계를 바탕으로 카메라 파라미터(내부 행렬, 왜곡 계수)를 추정합니다.
*   추정된 파라미터를 사용해 원본 이미지의 렌즈 왜곡을 보정하고 시각화합니다.

### 2. 전체 코드 (0312-1.py)
```python
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (25mm)
square_size = 25.0 

# 코너 정밀화 조건 설정 (오차 0.001 내지 최대 30회 반복)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 3D 좌표 공간 생성 (Z=0으로 가정, 모든 이미지에서 동일 격자 구조)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 실제 좌표(objpoints)와 이미지 코너 좌표(imgpoints)를 저장할 리스트
objpoints = [] 
imgpoints = [] 

# 읽어올 이미지 경로 설정
images = glob.glob("0312/images/calibration_images/left*.jpg")
img_size = None

# [1. 체크보드 코너 검출]
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_size is None:
        img_size = gray.shape[::-1]

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출 성공 시 처리
    if ret == True:
        objpoints.append(objp)
        
        # 코너 픽셀 위치 정밀화 (Sub-pixel accuracy)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# [2. 카메라 캘리브레이션]
# 행렬 K와 왜곡계수 dist를 계산합니다.
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:\n", K)
print("\nDistortion Coefficients:\n", dist)

# [3. 왜곡 보정 시각화]
test_img = cv2.imread(images[0])
undistorted_img = cv2.undistort(test_img, K, dist, None, K)

cv2.imshow('Original Image', test_img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. 요구사항 별 핵심 코드 설명
*   **체크보드 검출 및 대응 좌표 구성:**
    ```python
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
    ```
    > `findChessboardCorners`를 이용해 이미지 내 2D 코너를 찾습니다. 성공한 이미지에 대해서만 미리 25mm 간격으로 정의해둔 실제 3D 좌표(`objp`)와 매핑시킵니다.
*   **파라미터 추정 및 왜곡 보정:**
    ```python
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undistorted_img = cv2.undistort(test_img, K, dist, None, K)
    ```
    > 추출한 좌표를 `calibrateCamera`에 넣고 돌리면 카메라 행렬(`K`) 및 왜곡 계수(`dist`)를 얻을 수 있습니다. 얻은 파라미터를 `undistort`에 전달해 렌즈 왜곡에 의해 휘어진 부분을 반듯하게 폅니다.

### 4. 결과 사진
<!-- 여기에 캘리브레이션 결과 콘솔 메시지 캡처 및 전/후 이미지 캡처 삽입 -->
![Assignment 1 결과](images/1-res1.png)
![Assignment 1 결과 2](images/1-res2.png)

---

## 📌 과제 2: 이미지 Rotation & Transformation (`0312-2.py`)

### 1. 문제 정의
*   단일 이미지에 기하학적 변환(회전, 스케일링, 평행이동)을 동시에 적용합니다.
*   요구사항: 현재 위치 기준 **+30도 회전**, 크기 **0.8배 축소**, x축 **+80px**, y축 **-40px** 이동.

### 2. 전체 코드 (0312-2.py)
```python
import cv2
import numpy as np
import os
from pathlib import Path

# 테스트 이미지 로드 (절대 경로 보장)
base_dir = Path(__file__).parent
img_path = str(base_dir / "images" / "rose.png")
if not os.path.exists(img_path):
    img_path = str(base_dir / "images" / "calibration_images" / "left01.jpg")

img = cv2.imread(img_path)
h, w = img.shape[:2]

# 중심 좌표 및 변환 설정
center = (w / 2, h / 2)
angle = 30   # +30도 회전
scale = 0.8  # 0.8배 스케일링

# 회전+스케일 행렬 생성
M = cv2.getRotationMatrix2D(center, angle, scale)

# 평행 이동(translation) 적용 (x축 +80px, y축 -40px)
M[0, 2] += 80
M[1, 2] -= 40

# 아핀 변환 적용
transformed_img = cv2.warpAffine(img, M, (w, h))

# 결과 시각화
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. 요구사항 별 핵심 코드 설명
*   **회전 및 스케일 행렬 생성:**
    ```python
    M = cv2.getRotationMatrix2D(center, angle, scale)
    ```
    > 주어진 함수를 이용해 설정한 중심점 기반 `30`도 회전 및 `0.8` 크기로 변환 행렬 M을 생성합니다.
*   **평행이동 수동 편입 후 적용:**
    ```python
    M[0, 2] += 80
    M[1, 2] -= 40
    transformed_img = cv2.warpAffine(img, M, (w, h))
    ```
    > `getRotationMatrix2D`로 만든 $2 \times 3$ 행렬의 세 번째 열(인덱스 2)은 $X, Y$ 축 이동을 관장합니다. 여기에 각각 요구사항대로 $80, -40$을 더해준 후, `warpAffine`으로 이미지에 반영합니다.

### 4. 결과 사진
<!-- 여기에 Rotation & Transform 결과 변화 사진 캡처 삽입 -->
![Assignment 2 결과]()

---

## 📌 과제 3: Stereo Disparity 기반 Depth 추정 (`0312-3.py`)

### 1. 문제 정의
*   같은 장면을 바라보는 Left / Right 이미지 쌍을 이용해, 양안 시차(Disparity)를 계산합니다.
*   이를 바탕으로 피사체의 Depth를 환산하고, 지정된 개별 검출 객체(ROI) 중 제일 가까운 물체와 먼 물체를 판별합니다.

### 2. 전체 코드 (0312-3.py)
```python
import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent
output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 흑백 이미지 로딩
left_color = cv2.imread(str(base_dir / "images" / "left.png"))
right_color = cv2.imread(str(base_dir / "images" / "right.png"))
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

f, B = 700.0, 0.12 # Focal length / Baseline
rois = { "Painting": (55, 50, 130, 110), "Frog": (90, 265, 230, 95), "Teddy": (310, 35, 115, 90) }

# [1. Disparity 계산] float 변환 후 16 나누기
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity_16S = stereo.compute(left_gray, right_gray)
disparity = disparity_16S.astype(np.float32) / 16.0

# [2. Depth 계산] Z = fB / d
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity, dtype=np.float32)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# [3. ROI별 평균 추출]
results = {}
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    valid_roi_mask = roi_disp > 0
    mean_disp = np.mean(roi_disp[valid_roi_mask]) if np.any(valid_roi_mask) else 0
    mean_depth = np.mean(roi_depth[valid_roi_mask]) if np.any(valid_roi_mask) else float('inf')
    results[name] = {"disparity": mean_disp, "depth": mean_depth}

# [4. 결과 출력 및 제일 가깝고/먼 객체 탐색]
for name, res in results.items(): print(f"[{name}] Disparity: {res['disparity']:.2f}, Depth: {res['depth']:.4f}")
filtered = {k: v for k, v in results.items() if v['depth'] != float('inf')}
print(f"가장 가까운 객체: {min(filtered, key=lambda k: filtered[k]['depth'])}")
print(f"가장 먼 객체: {max(filtered, key=lambda k: filtered[k]['depth'])}")

# (시각화 설정 및 출력 코드는 원본의 5번~9번 로직으로 작동)
# 컬러맵 씌우기 및 cv2.imshow ...
```

### 3. 요구사항 별 핵심 코드 설명
*   **Disparity Map 계산 및 정규화:**
    ```python
    stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
    disparity_16S = stereo.compute(left_gray, right_gray)
    disparity = disparity_16S.astype(np.float32) / 16.0
    ```
    > `StereoBM_create`를 사용하여 양안 시차를 픽셀 단위로 환산합니다. 이때 OpenCV의 BM 알고리즘 구현상 결과가 `16배 스케일된 short 정수형`으로 반환되므로, 연산의 정확도를 위해 `16.0` 실수형으로 나눠주어야 요구사항에 맞는 정확한 disparity 맵이 나타납니다.
*   **Depth Map 변환 및 ROI 판별:**
    ```python
    valid_mask = disparity > 0
    depth_map[valid_mask] = (f * B) / disparity[valid_mask]
    ```
    > disparity가 `0`보다 클 때(비매칭 오류 방지)만 $Z = \frac{fB}{d}$ 공식을 대입하여 뎁스를 구합니다. 이후 `rois` 딕셔너리의 좌표 박스를 슬라이싱 하여 해당 픽셀들의 평균을 추출해 가까운 순서를 비교합니다. 계산 결과, 가장 거리가 짧은(Depth가 작은) 물체는 개구리(Frog)임을 판별했습니다.

### 4. 결과 사진
<!-- 여기에 Depth 결과물 콘솔 메세지 및 Disparity Map, Depth Map 캡처 삽입 -->
![Assignment 3 결과]()
