import cv2
import numpy as np
import os
from pathlib import Path

# =========================================================
# 1. 테스트 이미지 로드 및 경로 설정
# =========================================================
# Python 코드 파일이 있는 현재 위치를 기준으로 절대 경로를 설정합니다. 
# (에러 없이 안전하게 이미지를 불러오기 위함)
base_dir = Path(__file__).parent
img_path = str(base_dir / "images" / "rose.png")

# 만약 rose.png 파일을 찾을 수 없다면 첫번째 과제의 체크보드 이미지를 대신 사용합니다.
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found! Using left01.jpg instead.")
    img_path = str(base_dir / "images" / "calibration_images" / "left01.jpg")

# 이미지를 불러오고 이미지의 높이(h)와 너비(w) 정보를 가져옵니다.
img = cv2.imread(img_path)
h, w = img.shape[:2]

# =========================================================
# 2. 이미지 변환(회전, 스케일)을 위한 기하학 중심 설정
# =========================================================
# 이미지를 팽이 돌리듯 회전시킬 때 '어디를 기준으로 돌릴 것인가'를 정합니다. 
# (여기서는 정중앙)
center = (w / 2, h / 2)

# 과제 요구사항 세팅
angle = 30  # 반시계 방향으로 30도 돌리기 (+30)
scale = 0.8 # 전체 이미지 크기를 80% 사이즈로 줄이기 (0.8배)

# [핵심] 기준점, 각도, 축소 비율을 던져주면, 해당 변환을 수학적으로 수행해 줄 
# 2x3 크기의 아핀 변환 행렬(M)을 OpenCV가 계산해서 만들어줍니다.
M = cv2.getRotationMatrix2D(center, angle, scale)

# =========================================================
# 3. 평행 이동(Translation) 덧붙이기
# =========================================================
# 아핀 변환 행렬(M)의 맨 오른쪽 '끝 열'은 이미지를 '좌우/상하로 얼마나 밀어낼 건지'를 결정합니다.
# M = [[cos, -sin, X축 이동량(tx)],
#      [sin,  cos, Y축 이동량(ty)]]

# 요구사항: x축으로 +80px(오른쪽), y축으로 -40px(위쪽) 이동
M[0, 2] += 80  # X축 이동량 덮어쓰기
M[1, 2] -= 40  # Y축 이동량 덮어쓰기

# =========================================================
# 4. 이미지 변환 적용 (Warp Affine)
# =========================================================
# 완성된 변환 행렬(M)을 주물럭거릴 원본 사진(img)에 덮어씌워버립니다. 
# cv2.warpAffine은 넘겨진 행렬대로 픽셀들을 새로운 위치에 쫙 이동시켜 줍니다.
transformed_img = cv2.warpAffine(img, M, (w, h))

# =========================================================
# 5. 결과 화면 출력
# =========================================================
# 이미지가 너무 클 수 있으므로 화면에 띄울 땐 0.5배(절반)로 임시 축소해서 띄웁니다.
scale_preview = 0.5
img_preview = cv2.resize(img, (int(w * scale_preview), int(h * scale_preview)))
transformed_preview = cv2.resize(transformed_img, (int(w * scale_preview), int(h * scale_preview)))

# 원본 이미지 창 하나, 회전+축소+이동된 이미지 창 하나 각각 띄우기
cv2.imshow('Original Image (Before)', img_preview)
cv2.imshow('Transformed Image (After)', transformed_preview)

print("Rotation & Transformation 결과 시각화 중... 아무 키나 누르면 종료됩니다.")
cv2.waitKey(0)              # 사용자가 키보드를 누를 때까지 창을 닫지 않고 무한 대기
cv2.destroyAllWindows()     # 키보드가 눌리면 모든 이미지 창 끄기
