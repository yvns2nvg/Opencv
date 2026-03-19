import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. 이미지 로드 및 경로 설정
# =========================================================
base_dir = Path(__file__).parent
img_path = str(base_dir / "dabo.jpg")
img = cv2.imread(img_path)

if img is None:
    print(f"Error: 이미지를 찾을 수 없습니다: {img_path}")
    exit()

# 시각화를 위해 원본의 RGB 사본 생성
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_result = img_rgb.copy()

# =========================================================
# 2. Canny 에지 맵 생성
# =========================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny 함수를 사용해 에지 맵 생성 (threshold1=100, threshold2=200)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# =========================================================
# 3. 허프 변환(Hough Transform)을 이용한 직선 검출
# =========================================================
# 거리(rho), 각도(theta), 직선으로 판단할 임계값(threshold)과
# 최소 선 길이(minLineLength), 선 사이의 최대 허용 간격(maxLineGap) 설정
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

# =========================================================
# 4. 검출된 직선 그리기
# =========================================================
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 원본 이미지(RGB 사본) 위에 빨간색(255, 0, 0), 두께 2로 선 그리기
        cv2.line(img_result, (x1, y1), (x2, y2), (255, 0, 0), 2)

# =========================================================
# 5. 결과 시각화 (Matplotlib)
# =========================================================
plt.figure(figsize=(12, 6))

# 첫 번째 구획: 원본 이미지
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_rgb)
plt.axis('off')

# 두 번째 구획: 직선이 검출된 결과 이미지
plt.subplot(1, 2, 2)
plt.title('Hough Transform Line Detection')
plt.imshow(img_result)
plt.axis('off')

plt.tight_layout()
plt.show()
