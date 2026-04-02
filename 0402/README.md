# OpenCV Image Recognition 실습 과제 (0402)

---

## 과제 1: 간단한 이미지 분류기 구현 (`0402-1.py`)

### 1. 문제 정의
*   `MNIST` 데이터셋을 활용하여 손글씨 숫자(0~9)를 분류하는 간단한 이미지 분류기를 구현합니다.
*   다층 퍼셉트론(MLP 구조: `Dense` 레이어 사용) 모델을 설계하고, 모델 훈련 결과를 시각화하여 확인합니다.

### 2. 전체 코드 (0402-1.py)
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np

# 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 (정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 간단한 신경망 모델 구축
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 모델 훈련
print("---- 모델 훈련 시작 ----")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("---- 모델 훈련 완료 ----")

# 4. 모델 평가 및 결과 시각화
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# 결과 출력 (훈련 정확도 및 손실 그래프)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '0402-1.png'))
# plt.show()
```

### 3. 요구사항 별 핵심 코드 설명
*   **데이터셋 로드 및 전처리:**
    ```python
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    ```
    > `keras`에서 제공하는 MNIST 데이터를 불러온 후, 픽셀 값(0~255)을 0~1 사이로 정규화(Normalization)하여 모델이 더 빠르고 정확하게 학습할 수 있도록 합니다.
*   **신경망 모델 구축:**
    ```python
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    ```
    > 28x28 크기의 2차원 이미지를 1차원(784)으로 펴주는 `Flatten`을 사용하며, 이후 `relu` 활성화 함수를 통과하는 크기 128의 `Dense` 계층을 거칩니다. 다중 클래스 분류이므로 마지막엔 크기 10(0~9)의 `softmax` 출력을 반환합니다.

### 4. 결과 사진
![Assignment 1 결과](0402-1.png)

---

## 과제 2: CIFAR-10 데이터셋을 활용한 CNN 모델 구축 (`0402-2.py`)

### 1. 문제 정의
*   `CIFAR-10` 데이터셋(10개 클래스의 32x32 컬러 이미지)을 활용하여 이미지 분류를 수행하는 합성곱 신경망(CNN)을 구축합니다.
*   학습된 모델을 저장하거나, 테스트 이미지(`dog.jpg`)를 직접 모델에 입력하여 모델이 예상하는 예측값과 확률을 시각적으로 확인합니다.

### 2. 전체 코드 (0402-2.py)
```python
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

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. CNN 모델 구축 (개선된 모델)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 모델 훈련
print("---- CNN 모델 훈련 시작 ----")
history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_test, y_test))
print("---- CNN 모델 훈련 완료 ----")

# 4. 모델 평가 및 dog.jpg 예측 (Prediction)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# 테스트 이미지 로드 및 전처리
test_img = cv2.imread(img_path)
test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img_resized = cv2.resize(test_img_rgb, (32, 32))
input_img = np.expand_dims(test_img_resized, axis=0) / 255.0

# 5. 예측 결과 추론
predictions = model.predict(input_img)
predicted_class = class_names[np.argmax(predictions)]
predicted_prob = np.max(predictions)

# 결과 출력 (훈련 정확도 및 예측 이미지)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
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
# plt.show()
```

### 3. 요구사항 별 핵심 코드 설명
*   **합성곱 신경망(CNN) 층 구성:**
    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        ...
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    ```
    > 개선된 모델에서는 각 합성곱(`Conv2D`) 층에 `padding='same'`을 추가하여 공간 차원을 유지하고, 레이어 출력 정규화를 위해 `BatchNormalization`을 추가하여 안정적이고 빠른 학습을 돕습니다. 또한 각 블록 이후에 `Dropout`을 배치하여 과적합(Overfitting)을 최소화하였습니다. 최종적으로 평탄화(`Flatten`) 후 `Dense` 레이어를 통과하여 추출된 특징들로 각 클래스 확률을 추론합니다.
*   **테스트 이미지 예측 처리:**
    ```python
    test_img_resized = cv2.resize(test_img_rgb, (32, 32))
    input_img = np.expand_dims(test_img_resized, axis=0) / 255.0
    predictions = model.predict(input_img)
    ```
    > 모델 예측에 직접 임의의 이미지(`dog.jpg`)를 사용하기 위해, 이미지를 로컬에서 읽은 후 모델 학습에 사용된 크기와 포맷(`32x32`, `RGB`, `배치 차원 확장`, `픽셀 스케일 정규화`)을 똑같이 맞춰 입력 데이터 차원 형태(`[1, 32, 32, 3]`)로 가공해주었습니다. 이렇게 전처리된 데이터를 `predict`에 넣어 가장 추측 확률이 높은 클래스를 뽑아냅니다.

### 4. 결과 사진
![Assignment 2 결과](0402-2.png)
