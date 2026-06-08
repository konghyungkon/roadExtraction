"""
02_deepLearningAndTraining.py
Road Extraction Project - U-Net 모델 정의 및 학습
📌 역할
   1) 전처리된 데이터(.npy) 로드
   2) Dice Coefficient 평가 지표 정의 (Segmentation 필수)
   3) U-Net 모델 설계 (Functional API)
   4) 학습 (fit) 및 최적 모델 저장 (.h5 / .keras)
   5) 결과 그래프 시각화 (Loss, Accuracy, Dice Score)

   수업 때 사용하던 cat & dog 이진분류 코드(ai_exam09_2_cat_and_dog.py 스타일)
    - npy 로딩 → model 정의 → compile → fit → save 흐름

    - Dice + Accuracy로 평가하고
    - 학습 로그를 그래프로 보여주고
    - 최적 가중치를 .h5로 저장하는 최종 학습 스크립트다.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ============================================================
# 1. 데이터 로딩 (npy 파일 읽기)
#    - 01_dataProcess.py 에서 미리 만들어 둔 데이터 사용
# ============================================================
print("Step 1: Numpy 데이터 로딩 중...")

try:
    X_train = np.load('./binary_X_train.npy')
    Y_train = np.load('./binary_Y_train.npy')
    X_test  = np.load('./binary_X_test.npy')
    Y_test  = np.load('./binary_Y_test.npy')

    print(f" -> X_train shape: {X_train.shape}")
    print(f" -> Y_train shape: {Y_train.shape}")
    print(f" -> X_test  shape: {X_test.shape}")
    print(f" -> Y_test  shape: {Y_test.shape}")

    # 이미지 크기 자동 인식 (128, 128, 3)
    image_h = X_train.shape[1]
    image_w = X_train.shape[2]
    channels = X_train.shape[3]

except FileNotFoundError:
    print("❌ 에러: .npy 파일이 없습니다. 01_dataProcess.py를 먼저 실행해주세요!")
    exit()

# ============================================================
# 2. Dice 계수 정의 (세그멘테이션에서 자주 쓰는 지표)
#    도로처럼 얇은 객체를 찾을 때 정확도(Accuracy)보다 훨씬 믿을만함.
# ============================================================
def dice_coef(y_true, y_pred, smooth=1e-7):
    # 1차원으로 쫙 펴서 계산
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # 교집합(Intersection) 계산
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Dice 공식: (2 * 교집합) / (A + B)
    return (2. * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

# ============================================================
# 3. U-Net 모델 정의 (Functional API)
#    - 두 버전의 U-Net을 절충해서,
#      너무 무겁지 않으면서도 기본 U-Net 형태 유지
# ============================================================
def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # --- [수축 경로 (Encoder)] ---
    # 1층 (32)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # 2층 (64)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # 3층 (128)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # --- [바닥 (Bottleneck)] ---
    # 4층 (256) - 가장 압축된 정보
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    d4 = Dropout(0.5)(c4)  # 과적합 방지

    # --- [확장 경로 (Decoder)] ---
    # 3층으로 복귀
    u5 = UpSampling2D((2, 2))(d4)
    u5 = Conv2D(128, (2, 2), activation='relu', padding='same')(u5)  # 크기 맞추기용 Conv
    m5 = concatenate([u5, c3])  # ★ Skip Connection (c3와 합체)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(m5)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    # 2층으로 복귀
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(64, (2, 2), activation='relu', padding='same')(u6)
    m6 = concatenate([u6, c2])  # ★ c2와 합체
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    # 1층으로 복귀
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(32, (2, 2), activation='relu', padding='same')(u7)
    m7 = concatenate([u7, c1])  # ★ c1과 합체
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    # --- [출력층] ---
    # 0(배경) ~ 1(도로) 확률 출력 (Sigmoid)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=inputs, outputs=outputs, name="road_unet")
    return model

# ============================================================
# 4. 모델 생성 & 컴파일
# ============================================================
print("Step 2: U-Net 모델 생성 중...")

input_shape = (image_h, image_w, channels)  # 01_dataProcess에서 만든 패치 크기 사용
model = build_unet(input_shape)

# cat/dog 프로젝트 스타일 + Dice Metric 추가
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', dice_coef]
)
model.summary()

# ============================================================
# 5. 콜백 설정 (모델 저장 & 조기 종료)
#    - 첫 번째 버전의 ModelCheckpoint + 두 번째 버전의 EarlyStopping 패턴 결합
# ============================================================
checkpoint_path = './road_extraction_model.h5' # .h5로 저장 (가장 범용적)
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1) # 5번 참음
]

# ============================================================
# 6. 학습 (Training)
#    - PDF에서 말한 "틀려가면서 배우기" 단계
#    - train 중 10%는 validation으로 사용, X_test는 나중에 평가용으로 남겨둠
# ============================================================
print("\nStep 3: 학습 시작! (AI가 공부하는 중... ☕)")

history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=8,   # 메모리 부족하면 8로 줄이세요
    validation_split=0.1, # 모의고사 데이터
    callbacks=callbacks,
    verbose=1
)
print("✅ 학습 완료!")

# ============================================================
# 7. Test 데이터셋으로 최종 평가
# ============================================================
print("Step 4: Test 데이터셋 평가 중...")

test_loss, test_acc, test_dice = model.evaluate(X_test, Y_test, verbose=1)
print(f"\n📊 [Test 결과] loss={test_loss:.4f}, acc={test_acc:.4f}, dice={test_dice:.4f}\n")

# ============================================================
# 8. 최종 모델 저장 (.h5 - cat/dog 때와 동일한 형식)
# ============================================================
final_path = './road_unet_final.h5'
model.save(final_path)
print(f"💾 최종 모델 저장 완료: {final_path}")

# ============================================================
# 6. 결과 평가 및 시각화
#    (그래프 그리기)
# ============================================================
print("\nStep 4: 결과 그래프 그리는 중...")

plt.figure(figsize=(15, 5))

# (1) 정확도 (Accuracy) - 높을수록 좋음
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy (Higher is Better)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# (2) 손실 (Loss) - 낮을수록 좋음
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (Lower is Better)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# (3) Dice Score (도로 추출 성능) - 높을수록 좋음 (★핵심 지표)
plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coef'], label='Train Dice')
plt.plot(history.history['val_dice_coef'], label='Val Dice')
plt.title('Dice Score (Higher is Better)')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("모든 과정이 끝났습니다. 생성된 .h5 모델로 예측을 진행하세요!")