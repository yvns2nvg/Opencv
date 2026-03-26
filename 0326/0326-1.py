import cv2 as cv
import matplotlib.pyplot as plt
import os

# 현재 스크립트 파일 위치를 기준으로 이미지 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, 'mot_color70.jpg')

# 이미지 읽기 (BGR 형식을 RGB로 변환하여 matplotlib에서 제대로 출력되도록 함)
img = cv.imread(img_path)
if img is None:
    print(f"이미지를 불러올 수 없습니다: {img_path}")
    exit(1)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 요구사항: SIFT 객체 생성 (특징점이 너무 많지 않도록 nfeatures=500으로 제한)
sift = cv.SIFT_create(nfeatures=500)

# 요구사항: 특징점 검출 및 디스크립터 계산
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# 요구사항: 특징점 시각화 (크기와 방향 포함 옵션 설정)
img_keypoints = cv.drawKeypoints(
    img_rgb, 
    keypoints, 
    None, 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 결과 출력 (원본 이미지와 특징점 시각화 이미지 나란히 배치)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_keypoints)
plt.title('SIFT Keypoints (Rich)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '0326-1.png'))
# plt.show()
