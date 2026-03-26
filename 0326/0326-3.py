import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
# 요구사항: 샘플파일로 img1.jpg, img2.jpg, img3.jpg 중 2개를 선택하여 사용
img1_path = os.path.join(base_dir, 'img2.jpg')
img2_path = os.path.join(base_dir, 'img3.jpg')

# 1. 두 개의 이미지를 불러옴
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다.")
    exit(1)

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 및 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 3. BFMatcher 및 knnMatch 적용
bf = cv.BFMatcher(cv.NORM_L2)
# 힌트: knnMatch()로 두 개의 최근접 이웃을 구함
matches = bf.knnMatch(des1, des2, k=2)

# 좋은 매칭점 선별 (Ratio Test)
good_matches = []
# 힌트: 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    print("충분한 매칭점이 없습니다 (최소 4개 필요).")
    exit(1)

# 매칭 결과 (Matching Result) 이미지 저장/시각화
img_matches = cv.drawMatches(
    img1_rgb, kp1, 
    img2_rgb, kp2, 
    good_matches, None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 4. 호모그래피 계산을 위한 좌표 추출
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 파노라마 합성 시 음수 좌표(잘림) 현상을 방지하기 위해 
# 오른쪽 이미지(img2)를 왼쪽 이미지(img1)의 좌표계로 변환합니다.
# 따라서 Homography는 img2(dst_pts)에서 img1(src_pts)로 가는 변환 행렬을 찾습니다.
H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

# 5. 한 이미지를 변환하여 다른 이미지와 정렬 (Image Alignment)
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 힌트: cv.warpPerspective()를 사용할 때 출력 크기를 파노라마 크기 (w1+w2, max(h1,h2))로 설정
result_w = w1 + w2
result_h = max(h1, h2)

# img2를 img1 평면으로 변환
warped_img = cv.warpPerspective(img2_rgb, H, (result_w, result_h))

# Warped image 공간 위에 기준점(원점)인 img1의 내용을 덮어씌움
warped_img[0:h1, 0:w1] = img1_rgb

# 6. 결과 시각화 (변환된 이미지와 부분 매칭 결과 투명 겹침은 아니며, 두 이미지를 나란히 출력)
# 요구사항: 변환된 이미지 (Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_matches)
plt.title('Matching Result (Ratio Test = 0.7)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(warped_img)
plt.title('Image Alignment (Warped & Stitched)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '0326-3.png'))
# plt.show()
