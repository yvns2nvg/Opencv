import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# =========================================================
# 1. 이미지 로드 및 초기 설정
# =========================================================
base_dir = Path(__file__).parent
img_path = str(base_dir / "coffee cup.JPG")
img = cv2.imread(img_path)

if img is None:
    print(f"Error: 이미지를 찾을 수 없습니다: {img_path}")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================================================
# 2. GrabCut 사각형 범위 및 배경/전경 모델 초기화
# =========================================================
# 커피잔이 위치한 영역을 대략적인 사각형(x, y, width, height)으로 지정
# (이미지 크기가 1280x960이므로 커피잔을 덮을 영역으로 세팅)
rect = (300, 150, 700, 750) 

# GrabCut 내부 계산을 위한 빈 배열 초기화 (1x65 크기의 float64 형식)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 이미지와 동일한 크기의 0으로 채워진 마스크 생성
mask = np.zeros(img.shape[:2], np.uint8)

# =========================================================
# 3. GrabCut 알고리즘 실행
# =========================================================
# 지정한 사각형(rect)을 초기 영역으로 삼아 5번(iterCount) 반복 수행
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# =========================================================
# 4. 마스크 처리 배경 제거
# =========================================================
# 확실한 배경(GC_BGD: 0)과 아마 배경(GC_PR_BGD: 2)인 곳은 0으로, 
# 확실한 전경(1)과 아마 전경(3)인 곳은 1로 변경
mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')

# 원본 이미지에 최종 마스크(0과 1)를 곱해 배경을 검게 지워냄
result = img * mask2[:, :, np.newaxis]
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# =========================================================
# 5. 결과 시각화 (Matplotlib)
# =========================================================
plt.figure(figsize=(15, 6))

# 첫 번째 구획: 원본 이미지 (영역 표시 상자 포함)
plt.subplot(1, 3, 1)
plt.title('Original Image (with ROI)')
plt.imshow(img_rgb)
ax = plt.gca()
rect_patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor='g', facecolor='none')
ax.add_patch(rect_patch)
plt.axis('off')

# 두 번째 구획: 처리된 마스크 이미지
plt.subplot(1, 3, 2)
plt.title('GrabCut Mask')
# 마스크가 1인 곳(전경)을 하얗게 보이도록 시각화
plt.imshow(mask2 * 255, cmap='gray')
plt.axis('off')

# 세 번째 구획: 객체(전경) 추출 결과
plt.subplot(1, 3, 3)
plt.title('Segmented Object')
plt.imshow(result_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
