import cv2
import numpy as np
from pathlib import Path

# 현재 스크립트 위치 기준으로 경로 설정
base_dir = Path(__file__).parent
output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기 (base_dir 기준 절대 경로 활용)
left_path = base_dir / "images" / "left.png"
right_path = base_dir / "images" / "right.png"

left_color = cv2.imread(str(left_path))
right_color = cv2.imread(str(right_path))

if left_color is None or right_color is None:
    raise FileNotFoundError(f"좌/우 이미지를 찾지 못했습니다. 경로: {left_path}, {right_path}")

# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
# StereoBM이 16배 스케일링된 정수를 반환하므로 float32로 변환 후 16.0으로 나눔
disparity_16S = stereo.compute(left_gray, right_gray)
disparity = disparity_16S.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
# disparity가 0 이하인 부분은 유효하지 않으므로 mask 처리
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity, dtype=np.float32)

# Z = (f * B) / d
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}
for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # 해당 roi 안에서 valid한(>0) 값들만 추출
    valid_roi_mask = roi_disp > 0
    if np.any(valid_roi_mask):
        mean_disp = np.mean(roi_disp[valid_roi_mask])
        mean_depth = np.mean(roi_depth[valid_roi_mask])
    else:
        mean_disp = 0
        mean_depth = float('inf')
        
    results[name] = {"disparity": mean_disp, "depth": mean_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("==== ROI Analysis ====")
for name, res in results.items():
    print(f"[{name}]")
    print(f"  - Mean Disparity: {res['disparity']:.2f}")
    print(f"  - Mean Depth:     {res['depth']:.4f} m\n")

# 가장 가까운 물체와 먼 물체 찾기
filtered_results = {k: v for k, v in results.items() if v['depth'] != float('inf')}
if filtered_results:
    closest_roi = min(filtered_results, key=lambda k: filtered_results[k]['depth'])
    farthest_roi = max(filtered_results, key=lambda k: filtered_results[k]['depth'])
    print(f"가장 가까운 객체 (제일 Depth가 작음): {closest_roi}")
    print(f"가장 먼 객체 (제일 Depth가 큼):   {farthest_roi}")
else:
    print("유효한 연산 결과가 없습니다.")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)

# -----------------------------
# 9. 출력
# -----------------------------
scale_preview = 1.0
h_disp, w_disp = disparity_color.shape[:2]
disp_preview = cv2.resize(disparity_color, (int(w_disp*scale_preview), int(h_disp*scale_preview)))
depth_preview = cv2.resize(depth_color, (int(w_disp*scale_preview), int(h_disp*scale_preview)))
color_preview = cv2.resize(left_vis, (int(w_disp*scale_preview), int(h_disp*scale_preview)))

cv2.imshow('Left ROI', color_preview)
cv2.imshow('Disparity Map', disp_preview)
cv2.imshow('Depth Map', depth_preview)

print("\n시각화된 이미지를 띄웠습니다. 아무 키나 누르면 종료됩니다...")
cv2.waitKey(0)
cv2.destroyAllWindows()
