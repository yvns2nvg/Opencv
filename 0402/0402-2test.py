# TensorFlow 라이브러리를 tf라는 이름으로 불러옵니다.
# 딥러닝 모델 생성, 학습, 평가에 사용하는 핵심 라이브러리입니다.
import tensorflow as tf

# TensorFlow 안에 포함된 고수준 API인 Keras를 불러옵니다.
# 모델을 쉽게 구성할 수 있게 해줍니다.
from tensorflow import keras

# 신경망의 각 층(Conv2D, Dense, Dropout 등)을 만들기 위해 layers를 불러옵니다.
from tensorflow.keras import layers

# 수치 계산과 배열 처리를 위한 NumPy 라이브러리입니다.
import numpy as np

# 학습 결과 그래프와 이미지 출력을 위해 matplotlib를 불러옵니다.
import matplotlib.pyplot as plt

# 외부 이미지 파일(dog.jpg)을 열기 위해 PIL의 Image 모듈을 불러옵니다.
from PIL import Image

# 파일 존재 여부를 확인하기 위해 os 모듈을 불러옵니다.
import os


# CIFAR-10 데이터셋을 불러오기 전에 안내 문구를 출력합니다.
print("CIFAR-10 데이터셋 로드 중...")

# CIFAR-10 데이터셋을 로드합니다.
# x_train, y_train: 학습용 이미지와 정답
# x_test, y_test: 테스트용 이미지와 정답
# CIFAR-10은 32x32 크기의 컬러 이미지 10개 클래스로 구성된 데이터셋입니다.
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# 숫자 라벨(0~9)을 사람이 읽을 수 있는 클래스 이름으로 바꾸기 위한 리스트입니다.
# 예를 들어 예측값이 3이면 'cat'으로 해석할 수 있습니다.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# 이미지 픽셀 값을 0~255 범위에서 0~1 범위로 정규화합니다.
# 이렇게 하면 학습이 더 안정적으로 되고, 최적화가 잘 이루어집니다.
print("데이터 정규화 중...")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# CIFAR-10의 y값은 보통 (50000, 1), (10000, 1) 형태입니다.
# flatten()을 사용하면 (50000,), (10000,)처럼 1차원 배열로 바뀝니다.
# sparse_categorical_crossentropy를 사용할 때 이런 형태가 더 다루기 편합니다.
y_train = y_train.flatten()
y_test = y_test.flatten()


# 데이터의 전체 형태를 출력하여 제대로 로드되었는지 확인합니다.
print(f"훈련 데이터 형태: {x_train.shape}")
print(f"테스트 데이터 형태: {x_test.shape}")
print()


# CNN 모델을 만들기 시작한다는 안내 문구입니다.
print("CNN 모델 구축 중...")


# Sequential 모델은 층을 순서대로 쌓는 가장 기본적인 모델 구조입니다.
model = keras.Sequential([
    
    # 입력 이미지의 크기를 지정합니다.
    # CIFAR-10 이미지는 32x32 크기의 RGB 컬러 이미지이므로 shape=(32, 32, 3)입니다.
    layers.Input(shape=(32, 32, 3)),
    
    
    # -------------------- 첫 번째 Convolution Block --------------------
    
    # 32개의 필터를 사용하는 합성곱 층입니다.
    # (3,3) 크기의 필터로 이미지의 지역적 특징(모서리, 패턴 등)을 추출합니다.
    # padding='same'은 출력 이미지 크기를 입력과 같게 유지합니다.
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    
    # 배치 정규화(BatchNormalization)입니다.
    # 각 층의 출력 분포를 안정화하여 학습을 더 빠르고 안정적으로 만들어줍니다.
    layers.BatchNormalization(),
    
    # 같은 블록 안에서 한 번 더 합성곱을 수행해 특징을 더 깊게 추출합니다.
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    
    # 다시 배치 정규화를 적용합니다.
    layers.BatchNormalization(),
    
    # MaxPooling은 2x2 영역에서 가장 큰 값만 남겨 이미지 크기를 줄입니다.
    # 중요한 특징은 유지하면서 계산량을 줄이는 역할을 합니다.
    layers.MaxPooling2D((2, 2)),
    
    # Dropout은 일부 뉴런 출력을 무작위로 끊어 과적합을 방지합니다.
    # 여기서는 25%를 비활성화합니다.
    layers.Dropout(0.25),
    
    
    # -------------------- 두 번째 Convolution Block --------------------
    
    # 필터 수를 64개로 늘려 더 복잡한 특징을 추출합니다.
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 한 번 더 합성곱을 적용하여 특징 표현력을 높입니다.
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 다시 공간 크기를 줄입니다.
    layers.MaxPooling2D((2, 2)),
    
    # 과적합 방지를 위한 Dropout입니다.
    layers.Dropout(0.25),
    
    
    # -------------------- 세 번째 Convolution Block --------------------
    
    # 필터 수를 128개로 늘려 더 고수준의 특징을 추출합니다.
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 같은 블록 안에서 추가 합성곱을 수행합니다.
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    
    # 다시 MaxPooling으로 특징 맵의 크기를 줄입니다.
    layers.MaxPooling2D((2, 2)),
    
    # Dropout으로 과적합을 줄입니다.
    layers.Dropout(0.25),
    
    
    # -------------------- 분류기(Classifier) 부분 --------------------
    
    # 지금까지의 3차원 특징 맵을 1차원 벡터로 펼칩니다.
    # Dense 층은 1차원 입력을 받기 때문에 Flatten이 필요합니다.
    layers.Flatten(),
    
    # 완전연결층(Dense)입니다.
    # 앞에서 추출한 특징들을 종합해 분류에 적합한 표현으로 바꿉니다.
    layers.Dense(256, activation='relu'),
    
    # 분류기 부분에서도 Dropout을 크게 주어 과적합을 방지합니다.
    layers.Dropout(0.5),
    
    # 최종 출력층입니다.
    # CIFAR-10은 10개 클래스로 분류하므로 출력 뉴런도 10개입니다.
    # softmax를 사용해 각 클래스에 대한 확률값을 출력합니다.
    layers.Dense(10, activation='softmax')
])


# 모델의 학습 방법을 설정합니다.
model.compile(
    # Adam 옵티마이저를 사용합니다.
    # learning_rate=0.001은 가중치를 얼마나 빠르게 조정할지 결정하는 값입니다.
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    
    # 정답 라벨이 one-hot 인코딩이 아닌 정수 라벨이므로
    # sparse_categorical_crossentropy 손실 함수를 사용합니다.
    loss='sparse_categorical_crossentropy',
    
    # 학습 중 정확도도 함께 계산합니다.
    metrics=['accuracy']
)


# 모델의 전체 구조를 출력합니다.
# 각 층의 출력 형태와 학습해야 할 파라미터 개수를 확인할 수 있습니다.
print("모델 아키텍처:")
model.summary()
print()


# 실제 학습을 시작한다는 안내 문구입니다.
print("모델 훈련 중...")

# model.fit()은 학습 데이터를 이용해 모델을 훈련시키는 핵심 함수입니다.
history = model.fit(
    # 입력 이미지 데이터
    x_train,
    
    # 각 이미지에 대한 정답 라벨
    y_train,
    
    # 전체 데이터를 20번 반복 학습합니다.
    epochs=20,
    
    # 한 번에 64개 샘플씩 묶어서 학습합니다.
    batch_size=64,
    
    # 학습 데이터의 10%를 검증용 데이터로 자동 분리합니다.
    # 검증 데이터는 학습에는 사용하지 않고 성능 확인용으로만 씁니다.
    validation_split=0.1,
    
    # 학습 진행 상황을 출력합니다.
    verbose=1
)
print()


# 학습이 끝난 뒤, 테스트 세트에서 최종 성능을 평가합니다.
print("테스트 세트에서 성능 평가:")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# 정확도를 퍼센트 형태로 출력합니다.
print(f"테스트 정확도: {test_accuracy * 100:.2f}%")

# 손실값도 함께 출력합니다.
print(f"테스트 손실(Loss): {test_loss:.4f}")
print()


# 학습 과정에서 accuracy와 loss가 어떻게 변했는지 시각화합니다.
plt.figure(figsize=(12, 4))


# -------------------- 정확도 그래프 --------------------
plt.subplot(1, 2, 1)

# epoch별 학습 정확도
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)

# epoch별 검증 정확도
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)


# -------------------- 손실 그래프 --------------------
plt.subplot(1, 2, 2)

# epoch별 학습 손실
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)

# epoch별 검증 손실
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 그래프 간격을 자동 조정합니다.
plt.tight_layout()

# 그래프를 출력합니다.
plt.show()


# 테스트 이미지 일부에 대해 실제 예측 결과를 확인합니다.
print("테스트 이미지에 대한 예측:")

# 처음 10개 샘플만 확인합니다.
num_samples = 10

# 모델이 각 이미지에 대해 10개 클래스 확률을 예측합니다.
predictions = model.predict(x_test[:num_samples])


# 예측 결과를 이미지와 함께 시각화합니다.
plt.figure(figsize=(15, 3))
for i in range(num_samples):
    # 2행 5열 구조로 배치합니다.
    plt.subplot(2, 5, i + 1)
    
    # 테스트 이미지를 출력합니다.
    plt.imshow(x_test[i])
    
    # 예측 확률 중 가장 큰 값의 인덱스를 가져옵니다.
    # 즉, 모델이 가장 가능성이 높다고 판단한 클래스입니다.
    predicted_label = np.argmax(predictions[i])
    
    # 실제 정답 클래스입니다.
    true_label = y_test[i]
    
    # 예측이 맞으면 초록색, 틀리면 빨간색으로 제목 색을 설정합니다.
    color = 'green' if predicted_label == true_label else 'red'
    
    # 예측 클래스 이름과 실제 클래스 이름을 제목으로 표시합니다.
    plt.title(f'Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}',
              color=color, fontsize=9)
    
    # 축은 시각적으로 필요 없으므로 숨깁니다.
    plt.axis('off')

plt.tight_layout()
plt.show()


# -------------------- 외부 이미지(dog.jpg) 예측 부분 --------------------

# 예측할 외부 이미지 파일 이름입니다.
dog_image_path = 'dog.jpg'

# 해당 파일이 실제로 존재하는지 확인합니다.
if os.path.exists(dog_image_path):
    print(f"\ndog.jpg에 대한 예측:")
    
    # 이미지를 열고 RGB 형식으로 변환합니다.
    # convert('RGB')를 하는 이유는 모델 입력 형식을 CIFAR-10과 맞추기 위해서입니다.
    dog_img = Image.open(dog_image_path).convert('RGB')
    
    # 모델 입력 크기와 맞추기 위해 이미지를 32x32로 리사이즈합니다.
    dog_img_resized = dog_img.resize((32, 32))
    
    # 이미지를 NumPy 배열로 변환하고 0~1 범위로 정규화합니다.
    dog_array = np.array(dog_img_resized).astype("float32") / 255.0
    
    # 모델 입력은 (배치크기, 높이, 너비, 채널) 형태여야 하므로
    # 이미지 1장을 배치 형태로 만들기 위해 차원을 하나 추가합니다.
    # 결과 shape는 (1, 32, 32, 3)이 됩니다.
    dog_batch = np.expand_dims(dog_array, axis=0)
    
    # 외부 이미지에 대해 예측을 수행합니다.
    dog_prediction = model.predict(dog_batch)
    
    # 가장 확률이 높은 클래스 인덱스를 가져옵니다.
    predicted_class = np.argmax(dog_prediction[0])
    
    # 가장 높은 확률값 자체를 신뢰도로 사용합니다.
    confidence = np.max(dog_prediction[0])
    
    print(f"예측 클래스: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence * 100:.2f}%")
    
    # 예측 결과와 클래스별 확률 분포를 시각화합니다.
    plt.figure(figsize=(12, 5))
    
    # 왼쪽: 입력 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(dog_img_resized)
    plt.title(f'Input Image: {class_names[predicted_class]}')
    plt.axis('off')
    
    # 오른쪽: 각 클래스별 예측 확률
    plt.subplot(1, 2, 2)
    plt.barh(class_names, dog_prediction[0])
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.show()

else:
    # 파일이 없으면 예측을 건너뜁니다.
    print(f"\n경고: {dog_image_path} 파일을 찾을 수 없습니다.")
    print("외부 이미지 테스트를 건너뜁니다.")