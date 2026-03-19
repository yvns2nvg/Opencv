import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. 테스트 이미지 로드 및 경로 설정
# =========================================================
base_dir = Path(__file__).parent
img_path = str(base_dir / "edgeDetectionImage.jpg")
img = cv2.imread(img_path)

if img is None:
    print(f"Error: 이미지를 찾을 수 없습니다: {img_path}")
    exit()

# =========================================================
# 2. 그레이스케일 변환 및 에지 검출
# =========================================================
# 원본 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel 필터를 사용하여 x축과 y축 방향의 에지 검출 (ksize=3)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# =========================================================
# 3. 에지 강도 계산
# =========================================================
# 검출된 x, y축 에지를 바탕으로 전체 에지 강도(Magnitude) 계산
magnitude = cv2.magnitude(sobel_x, sobel_y)

# 시각화를 위해 계산된 에지 강도 배열을 8비트 정수형(uint8)으로 변환
magnitude = cv2.convertScaleAbs(magnitude)

# =========================================================
# 4. 결과 시각화 (Matplotlib)
# =========================================================
plt.figure(figsize=(12, 6))

# 첫 번째 구획: 원본 이미지
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 두 번째 구획: 에지 강도(Magnitude) 이미지
plt.subplot(1, 2, 2)
plt.title('Sobel Edge Magnitude')
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
