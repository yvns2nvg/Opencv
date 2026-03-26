import cv2 as cv
import matplotlib.pyplot as plt
import os

# 스크립트 파일 위치 기준 이미지 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
# mot_color80.jpg 라고 되어있으나 실제 제공된 파일명은 mot_color83.jpg 로 가정
img1_path = os.path.join(base_dir, 'mot_color70.jpg')
img2_path = os.path.join(base_dir, 'mot_color83.jpg')

# 요구사항: 1. cv.imread()를 사용하여 두 개의 이미지를 불러옴
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

if img1 is None or img2 is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
    exit(1)

# BGR -> RGB 및 BGR -> GRAY 변환
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 요구사항: 2. cv.SIFT_create()를 사용하여 특징점 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 요구사항: 3. cv.BFMatcher() 또는 cv.FlannBasedMatcher()를 사용하여 두 영상 간 특징점 매칭
# 힌트: BFMatcher(cv.NORM_L2, crossCheck=True)를 사용하면 간단한 매칭 가능
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 거리에 따라 정렬하여 우수한 매칭점들을 선별
matches = sorted(matches, key=lambda x: x.distance)

# 요구사항: 4. cv.drawMatches()를 사용하여 매칭 결과 시각화
# 상위 50개의 매칭점만 시각화 (가시성을 위함)
img_matches = cv.drawMatches(
    img1_rgb, kp1, 
    img2_rgb, kp2, 
    matches[:50], None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 요구사항: 5. matplotlib을 이용하여 매칭 결과 출력
plt.figure(figsize=(15, 7))
plt.imshow(img_matches)
plt.title('SIFT Feature Matching (BFMatcher)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '0326-2.png'))
# plt.show()
