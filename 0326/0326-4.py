import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
# 요구사항: 0326-3의 연장선, img2.jpg와 img3.jpg 사용
img1_path = os.path.join(base_dir, 'img2.jpg')
img2_path = os.path.join(base_dir, 'img3.jpg')

img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    exit(1)

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성 및 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFMatcher 및 knnMatch 적용
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio Test로 좋은 매칭점 선별
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    print("충분한 매칭점이 없습니다.")
    exit(1)

# 호모그래피 좌표 계산
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 파노라마 합성 (음수 좌표 방지를 위해 img2를 img1 좌표계로 변환)
H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

result_w = w1 + w2
result_h = max(h1, h2)

# 1. 오른쪽 이미지(img2)를 원근 변환
warped_img2 = cv.warpPerspective(img2_rgb, H, (result_w, result_h))

# 2. 왼쪽 이미지(img1)를 배치할 캔버스 생성
warped_img1 = np.zeros((result_h, result_w, 3), dtype=np.uint8)
warped_img1[0:h1, 0:w1] = img1_rgb

# 🌟 심화 경계 지우기 알고리즘 (Distance Transform 기반 Feathering Blending) 🌟
# 단순 평균이 아닌, 사진 중심부에 가까울수록 가중치를 더 주는 그라데이션 방식 도입
mask1_gray = cv.cvtColor(warped_img1, cv.COLOR_RGB2GRAY)
mask2_gray = cv.cvtColor(warped_img2, cv.COLOR_RGB2GRAY)

mask1 = (mask1_gray > 0).astype(np.uint8)
mask2 = (mask2_gray > 0).astype(np.uint8)

# 이미지의 가장자리에서부터 안쪽으로 거리를 측정 (경계는 0, 안쪽은 값이 커짐)
dist1 = cv.distanceTransform(mask1, cv.DIST_L2, 5)
dist2 = cv.distanceTransform(mask2, cv.DIST_L2, 5)

dist_sum = dist1 + dist2
dist_sum[dist_sum == 0] = 1e-6 # 0 나누기 방지

# 픽셀마다 투명도(가중치) 비율 맵 생성: 경계에서 서서히 0.0 ~ 1.0으로 변환됨
alpha1 = dist1 / dist_sum
alpha2 = dist2 / dist_sum

# 3채널(RGB) 컬러 영상과 곱하기 위해 축 늘리기
alpha1 = cv.merge([alpha1, alpha1, alpha1])
alpha2 = cv.merge([alpha2, alpha2, alpha2])

# 최종 점진적 크로스-블렌딩
blended_img = (warped_img1.astype(np.float32) * alpha1 + warped_img2.astype(np.float32) * alpha2).astype(np.uint8)

# 결과 출력
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
# 기존과 같은 날카로운 덮어쓰기 결과 (비교용)
sharp_result = warped_img2.copy()
sharp_result[0:h1, 0:w1] = img1_rgb
plt.imshow(sharp_result)
plt.title('Before (Sharp Boundary)')
plt.axis('off')

plt.subplot(1, 2, 2)
# 색상을 평균내어 경계를 없앤 블렌딩 결과
plt.imshow(blended_img)
plt.title('After (Average Blending)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '0326-4.png'))
plt.show()
