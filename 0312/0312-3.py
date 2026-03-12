import cv2
import numpy as np
from pathlib import Path

# =========================================================
# [초기 설정] 경로, 파라미터 및 관심 영역(ROI) 지정
# =========================================================
# 이 스크립트(.py)가 위치한 폴더를 기준으로 상대 경로의 꼬임을 막기 위해 절대경로(`base_dir`)를 잡습니다.
base_dir = Path(__file__).parent
output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True) # outputs 폴더가 없으면 새로 만듭니다.

# 좌/우 시점의 카메라가 찍은 스테레오 이미지를 흑백으로 불러옵니다. (Depth 연산은 흑백 픽셀 값 차이로 계산)
left_color = cv2.imread(str(base_dir / "images" / "left.png"))
right_color = cv2.imread(str(base_dir / "images" / "right.png"))

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다. 경로를 확인해주세요.")

# 컬러 화면위에 글씨나 네모 박스를 치기 위해 원본을 복사해둡니다.
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# 카메마 내장 스펙 정보 (focal length: 초점거리, B: 양 카메라 렌즈 간의 거리)
f = 700.0
B = 0.12

# 내가 거리(Depth)를 측정하고 싶은 특정 물체들의 위치 좌표(x, y, 가로길이, 세로길이) 미리 정의해두기
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# =========================================================
# 1. 시차(Disparity) 맵 계산
# =========================================================
# OpenCV에 내장된 스테레오 매칭 알고리즘 세팅 (블록 사이즈 15x15 구역 단위로 비교)
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

# 좌/우 흑백 이미지를 넣어서, 각 픽셀별로 화면 상 거리가 얼마나 차이나는지 (Disparity) 찾기
# (단, StereoBM은 기본적으로 결과값을 소수점 없이 표현하기 위해 16배 뻥튀기(16 Scale)해서 반환해줍니다)
disparity_16S = stereo.compute(left_gray, right_gray)

# 실제 정확한 픽셀 단위 시차값을 구하려면, float(소수)로 바꾸고 16.0으로 다시 나눠줘야 합니다.
disparity = disparity_16S.astype(np.float32) / 16.0


# =========================================================
# 2. 실제 깊이(Depth) 맵 변환  (공식: Z = fB / d)
# =========================================================
# 시차(disparity)가 0 이하라면 카메라가 매칭점을 찾지 못했다는 뜻이므로 거리를 구하지 않습니다.
valid_mask = disparity > 0

# 깊이값을 저장할 빈 공간(도화지) 준비
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 유효한(>0) 픽셀들에 한해서만 "초점거리 * 베이스라인 / 시차" 공식을 적용하여 실제 미터(m) 단위 깊이를 도출해냅니다!
depth_map[valid_mask] = (f * B) / disparity[valid_mask]


# =========================================================
# 3. 각 물체(ROI) 별 평균 시차(Disparity) 및 깊이(Depth) 분석
# =========================================================
results = {}

for name, (x, y, w, h) in rois.items():
    # 앞서 구한 전체 지도(map)에서 그 물체에 해당하는 네모 영역만 칼로 잘라옵니다. (슬라이싱)
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # 잘라온 영역 중에서 계산에 성공한 픽셀(값이 0보다 큼)들만 뽑아냅니다.
    valid_roi_mask = roi_disp > 0
    
    if np.any(valid_roi_mask):
        # 성공한 픽셀들의 평균값 도출
        mean_disp = np.mean(roi_disp[valid_roi_mask])
        mean_depth = np.mean(roi_depth[valid_roi_mask])
    else:
        # 실패했다면 깊이는 무한대(알 수 없음) 취급
        mean_disp = 0
        mean_depth = float('inf')
        
    # 물체 이름별로 결과를 딕셔너리에 갈무리 보관
    results[name] = {"disparity": mean_disp, "depth": mean_depth}


# =========================================================
# 4. 콘솔에 분석 결과 예쁘게 텍스트로 치기
# =========================================================
print("==== ROI Analysis (각 물체별 깊이 분석 결과) ====")
for name, res in results.items():
    print(f"[{name}]")
    print(f"  - 평균 시차(Disparity): {res['disparity']:.2f} 픽셀")
    print(f"  - 평균 깊이(Depth)    : {res['depth']:.4f} m\n")

# 깊이를 측정 성공한 물체들 사이에서 누가 카메라에 제일 가깝고 누가 제일 먼지 정렬해서 비교해봅니다.
filtered_results = {k: v for k, v in results.items() if v['depth'] != float('inf')}
if filtered_results:
    # min은 값이 가장 작은 것(=제일 가까운 것), max는 값이 가장 큰 것(=제일 멀리 있는 것)
    closest_roi = min(filtered_results, key=lambda k: filtered_results[k]['depth'])
    farthest_roi = max(filtered_results, key=lambda k: filtered_results[k]['depth'])
    print(f"🎯 가장 가까운 객체 (Depth 값이 최저치임): {closest_roi}")
    print(f"🔭 가장 먼 객체 (Depth 값이 최고치임):   {farthest_roi}")


# =========================================================
# [부록] 5~8번. 결과 이미지를 그라데이션 컬러로 시각화 (눈으로 편하게 보기 위함)
# =========================================================
# 5. 시차(Disparity) 이미지 색깔 먹이기 (가까울수록 빨강, 멀수록 파랑)
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan
d_min, d_max = np.nanpercentile(disp_tmp, 5), np.nanpercentile(disp_tmp, 95)
if d_max <= d_min: d_max = d_min + 1e-6
disp_scaled = np.clip((disp_tmp - d_min) / (d_max - d_min), 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)
# 일반 흑백이 아닌 푸른색부터 붉은색까지 예쁘게 퍼지는 컬러맵 입히기 (JET 옵션)
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# 6. 깊이(Depth) 이미지 색깔 먹이기 (가까울수록 빨강, 멀수록 파랑이 되도록 값 반전시킴)
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]
    z_min, z_max = np.percentile(depth_valid, 5), np.percentile(depth_valid, 95)
    if z_max <= z_min: z_max = z_min + 1e-6
    depth_scaled = np.clip((depth_map - z_min) / (z_max - z_min), 0, 1)
    # 시차와 다르게 거리는 클수록 먼 것이므로 1.0에서 빼서 색 배합을 뒤집어줍니다.
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# 7. 원본 좌측/우측 이미지에 관심구역(ROI) 네모 상자와 이름 치기
left_vis = left_color.copy()
for name, (x, y, w, h) in rois.items():
    # 초록색(0,255,0) 네모 박스 그리기
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 그 네모 위쪽에 이름(텍스트) 띄우기
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 8. 만들어낸 그림들을 파일 폴더로 저장
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)

# 9. 화면에 결과물 팝업 창 띄우기 (만약 너무 크면 scale_preview를 조절하세요)
scale_preview = 1.0
h_disp, w_disp = disparity_color.shape[:2]
disp_preview = cv2.resize(disparity_color, (int(w_disp*scale_preview), int(h_disp*scale_preview)))
depth_preview = cv2.resize(depth_color, (int(w_disp*scale_preview), int(h_disp*scale_preview)))
color_preview = cv2.resize(left_vis, (int(w_disp*scale_preview), int(h_disp*scale_preview)))

cv2.imshow('Left Original (ROI)', color_preview)
cv2.imshow('Calculated Disparity Map', disp_preview)
cv2.imshow('Calculated Depth Map', depth_preview)

print("\n시각화된 이미지를 띄웠습니다. 아무 키나 누르면 코드 종료.")
cv2.waitKey(0)         # 키보드를 누를 때까지 창 가만히 냅두기
cv2.destroyAllWindows()# 키보드 눌리면 창 모두 없애기
