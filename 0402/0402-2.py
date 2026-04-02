import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, 'dog.jpg')

# 1. CIFAR-10 데이터셋 로드
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리 (정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 클래스 이름 정의
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. CNN 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 모델 훈련
print("---- CNN 모델 훈련 시작 ----")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("---- CNN 모델 훈련 완료 ----")

# 4. 모델 평가 및 dog.jpg 테스트 예측 진행
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# 테스트 이미지 예측
test_img = cv2.imread(img_path)
if test_img is None:
    print(f"이미지를 불러올 수 없습니다: {img_path}")
    exit(1)

test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# CIFAR-10 입력 크기에 맞게 리사이즈 (32x32)
test_img_resized = cv2.resize(test_img_rgb, (32, 32))

# 모델 입력 형태로 변환 (배치 차원 추가 및 정규화)
input_img = np.expand_dims(test_img_resized, axis=0) / 255.0
predictions = model.predict(input_img)
predicted_class = class_names[np.argmax(predictions)]
predicted_prob = np.max(predictions)

# 결과 출력 (훈련 정확도 및 테스트 이미지 예측 결과)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('CNN Model Accuracy')

plt.subplot(1, 2, 2)
plt.imshow(test_img_rgb)
plt.title(f"Prediction: {predicted_class} ({predicted_prob*100:.2f}%)")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '0402-2.png'))
print(f"결과 이미지 저장 완료: {os.path.join(base_dir, '0402-2.png')}")
