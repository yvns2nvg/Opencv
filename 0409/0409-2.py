import cv2
import mediapipe as mp
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # MediaPipe FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    # OpenCV 웹캠 캡처 시작
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("MediaPipe FaceMesh 실시간 인식을 시작합니다. (종료하려면 ESC)")

    frame_idx = 0
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오 프레임을 읽을 수 없습니다.")
            break
            
        # 좌우 반전 시켜서 거울 모드로 보여줌 (선택사항)
        # frame = cv2.flip(frame, 1)

        # BGR을 RGB로 변환하여 MediaPipe에 전달
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # 얼굴 랜드마크가 검출된 경우
        if results.multi_face_landmarks:
            frame_idx += 1
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = frame.shape
                # 468개의 랜드마크 좌표를 화면 크기에 맞게 변환하여 점으로 시각화
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # 웹캠 웜업을 고려하여 30번째 정상 검출 윈도우 프레임을 README 캡처용으로 자동 저장
            if frame_idx == 30 and not saved:
                save_path = os.path.join(base_dir, "0409-2.png")
                cv2.imwrite(save_path, frame)
                print(f"30번째 프레임이 랜드마크 캡처로 저장되었습니다: {save_path}")
                saved = True

        cv2.imshow('MediaPipe FaceMesh (468 Landmarks)', frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
