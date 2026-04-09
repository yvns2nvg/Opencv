import cv2
import numpy as np
import os
import sys

# sort.py에서 클래스 가져오기
from sort import Sort

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights = os.path.join(base_dir, "L06", "yolov3.weights")
    model_cfg = os.path.join(base_dir, "L06", "yolov3.cfg")
    video_path = os.path.join(base_dir, "L06", "slow_traffic_small.mp4")
    classes_path = os.path.join(base_dir, "coco.names")

    # COCO 클래스 이름 로드
    classes = []
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]

    # YOLO 모델 로드
    net = cv2.dnn.readNet(model_weights, model_cfg)
    layer_names = net.getLayerNames()
    try:
        output_layers_indices = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]
    except:
        output_layers = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        sys.exit()

    # VideoWriter 설정
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0 # 기본값
    out_video_path = os.path.join(base_dir, "0409-1_output.mp4")
    # GitHub 등 웹에서 바로 재생하려면 avc1 (H.264) 코덱 사용이 필요하여 수정
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # SORT 추적기 초기화
    tracker = Sort()
    frame_idx = 0

    print("YOLO + SORT 비디오 추적을 시작합니다. (종료하려면 ESC)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오 재생 종료.")
            break
        
        frame_idx += 1
        height, width, channels = frame.shape
        
        # YOLO 이미지 전처리 및 추론
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        # 출력 결과 분석 (Bounding box, 예측 클래스 및 확률)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # 임계값: 0.5 이상인 경우만 취급
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-Maximum Suppression (중복 박스 제거)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # SORT 입력을 위한 Detection 포맷 맞추기: [x1, y1, x2, y2, score]
        dets = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                dets.append([x, y, x + w, y + h, confidences[i]])
        
        dets = np.array(dets)
        if len(dets) == 0:
            dets = np.empty((0, 5))
            
        # SORT 업데이트 및 Track ID 할당
        trackers = tracker.update(dets)
        
        # 추적된 객체들을 프레임에 시각화
        for d in trackers:
            dx1, dy1, dx2, dy2, track_id = map(int, d)
            
            # 박스 그리기
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            
            # ID 정보 표시
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # 프레임 시각화 및 영상 저장
        cv2.imshow("Multi-Object Tracking (SORT)", frame)
        video_writer.write(frame)
        
        # 특정 프레임(약 50프레임)에서 결과 캡처하여 README 용 저장
        if frame_idx == 50:
            save_path = os.path.join(base_dir, "0409-1.png")
            cv2.imwrite(save_path, frame)
            print(f"50번째 프레임이 저장되었습니다: {save_path}")
            
        if cv2.waitKey(1) == 27: # ESC
            break
            
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
