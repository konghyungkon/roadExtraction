"""
01_dataProcess.py

Road Extraction Project - 데이터 전처리 스크립트

📌 이 파일의 역할 (수업 시간 코드와의 연결)

1) 이미지/마스크 로딩 방식
   - ai_exam09_1_image_convert.py에서 했던 것처럼
     glob으로 파일 경로를 모두 가져온 뒤, for문으로 하나씩 처리한다.

2) 전처리 공통점
   - 이미지 리사이즈 → (image_w, image_h)
   - 이미지 스케일링 → 0~255 → 0~1 로 나누기 (X = X / 255.0)

3) 세그멘테이션이라서 달라지는 점 (가장 중요한 부분)
   - Y(라벨)가 숫자(0, 1 클래스 인덱스)가 아니라
     "도로만 흰색으로 칠해진 흑백 이미지"이다.
   - 그래서 Y에는
       (H, W) 크기의 흑백 이미지를
       (H, W, 1) 형태의 0/1 mask 로 저장한다.

4) 학습/테스트 분리
   - heart disease, titanic, cat/dog 전부 그랬듯이
     train_test_split() 써서 train / test 를 나눈다.
   - 여기서는 train + val 폴더를 모두 모아서 한 번에 섞은 뒤
     그 중 10%를 test 로 쓴다.
"""

from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import os

# -------------------------------------------------------------------
# 1. 기본 설정 (경로, 이미지 크기, 리스트 초기화)
# -------------------------------------------------------------------

# 데이터가 들어있는 루트 폴더
# └─ archive/tiff/
#       ├─ train/
#       ├─ train_labels/
#       ├─ val/
#       └─ val_labels/
base_dir = './archive/tiff/'
# train 과 val 둘 다 한 번에 모아서 나중에 train/test로 나눌 거임
categories = ['train', 'val']
patch_size = 100  # 자를 크기 (가로, 세로)
step = 100        # 겹치지 않고 자르려면 patch_size와 동일하게 설정

X = []  # 원본 이미지(문제) 담을 리스트
Y = []  # 정답 마스크(답안) 담을 리스트
files = None

# 🔥 메모리 보호용 상한
#MAX_PATCHES = 10000   # 일단 1만 개까지만 사용 (나중에 늘려도 됨)
# 메모리 터짐 방지

# 학습시킬 폴더 목록
# Kaggle 데이터셋은 train과 val 폴더가 나뉘어 있는데,
# 데이터를 최대한 많이 쓰기 위해 둘 다 가져와서 합칠 예정입니다.
print("데이터 전처리 시작><!!")

# -------------------------------------------------------------------
# 2. train / val 폴더를 돌면서 이미지 & 마스크 읽어오기
#    - ai_exam09_1_image_convert.py의 구조를 그대로 확장한 형태
# -------------------------------------------------------------------
for folder in categories:
    # 경로 설정 (문자열 더하기 방식 - 수업 스타일)
    img_dir = base_dir + folder + '/'  # 예: ./archive/tiff/train/
    mask_dir = base_dir + folder + '_labels/'  # 예: ./archive/tiff/train_labels/

    # 파일 목록 가져오기
    img_files = glob.glob(img_dir + '*.tiff')
    mask_files = glob.glob(mask_dir + '*.tif')

    # glob으로 읽으면 순서가 보장되지 않으므로 반드시 정렬 필요
    # 정렬해야 같은 번호의 이미지와 마스크가 매칭됨
    img_files = sorted(img_files)
    mask_files = sorted(mask_files)

    # 개수 확인
    if len(img_files) != len(mask_files):
        print("경고: 이미지와 마스크 개수가 다릅니다!")

    # 파일 하나씩 꺼내서 처리 (enumerate 사용)
    for i, (img_f, mask_f) in enumerate(zip(img_files, mask_files)):
        try:
            # 1) 원본 열기 (리사이즈 안 함!)
            img = Image.open(img_f).convert('RGB')
            mask = Image.open(mask_f).convert('L') # 흑백으로

            # 원본 크기 가져오기
            w, h = img.size

            # 2) 슬라이딩 윈도우 방식으로 타일링
            for y in range(0, h - patch_size + 1, step):
                for x in range(0, w - patch_size + 1, step):
                    # 타일 영역 설정
                    box = (x, y, x + patch_size, y + patch_size)
                    # 이미지와 마스크 자르기
                    img_patch = img.crop(box)
                    mask_patch = mask.crop(box)
                    # 크기 검증 (혹시 모를 경계 오류 방지)
                    if img_patch.size != (patch_size, patch_size):
                        continue

                    # --- 이미지 저장 ----
                    img_arr = np.asarray(img_patch)  # (256, 256, 3)
                    X.append(img_arr)

                    # --- 마스크 저장 ---
                    mask_arr = np.asarray(mask_patch)  # (256, 256)
                    mask_arr = mask_arr / 255.0  # 0~1 정규화
                    mask_arr = (mask_arr > 0.5).astype(int)  # 이진화
                    mask_arr = mask_arr.reshape(patch_size, patch_size, 1)  # (256, 256, 1)
                    Y.append(mask_arr)

            # 진행 상황 출력
            if i % 10 == 0:
                print(f"{folder} : {i}번째 이미지 타일링 완료 → 누적 {len(X)}개")

        except:
            print("에러 발생")

# -------------------------------------------------------------------
# 3. Numpy 변환 및 저장
# -------------------------------------------------------------------
print("Numpy 배열로 변환 중... (메모리 주의)")
X = np.array(X)
Y = np.array(Y)

print("최종 생성된 패치 개수:", len(X))

# 학습용 데이터 정규화 (float32 변환)
# 메모리가 부족하면 이 부분에서 터질 수 있으니 주의!
X = X/255

print("학습/테스트 데이터 분리 중...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=777)

print(f"저장 중... (Train: {len(X_train)}, Test: {len(X_test)})")
np.save('./binary_X_train.npy', X_train)
np.save('./binary_X_test.npy', X_test)
np.save('./binary_Y_train.npy', Y_train)
np.save('./binary_Y_test.npy', Y_test)

print("전처리 끝! 02_learning.py 실행하러 가자!")