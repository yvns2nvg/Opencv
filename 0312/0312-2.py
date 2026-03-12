import cv2
import numpy as np
import os
from pathlib import Path

# 1. 테스트 이미지 로드 (과제 2용 이미지, 예: rose.png)
base_dir = Path(__file__).parent
img_path = str(base_dir / "images" / "rose.png")
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found! Using left01.jpg instead.")
    img_path = str(base_dir / "images" / "calibration_images" / "left01.jpg")

img = cv2.imread(img_path)
h, w = img.shape[:2]

# 2. 중심 좌표, 회전 각도, 스케일 설정
center = (w / 2, h / 2)
angle = 30  # +30도 회전
scale = 0.8  # 0.8배 크기 조정

# 3. 회전 행렬 생성 (회전 & 크기 조절)
M = cv2.getRotationMatrix2D(center, angle, scale)

# 4. 평행 이동 적용 (x축 +80px, y축 -40px)
# M의 마지막 열이 평행 이동(translation)을 담당합니다.
# M = [[cos, -sin, tx],
#      [sin,  cos, ty]]
M[0, 2] += 80
M[1, 2] -= 40

# 5. 아핀 변환 적용
transformed_img = cv2.warpAffine(img, M, (w, h))

# 6. 결과 출력
cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', transformed_img)

print("Rotation & Transformation 결과 시각화 중... 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)
cv2.destroyAllWindows()
