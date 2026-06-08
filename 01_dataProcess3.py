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
patch_size = 128  # 자를 크기 (가로, 세로)
step = 128        # 겹치지 않고 자르려면 patch_size와 동일하게 설정

X = []  # 원본 이미지(문제) 담을 리스트
Y = []  # 정답 마스크(답안) 담을 리스트
files = None

# [핵심] 눈으로 확인할 사진들 저장할 폴더 만들기
save_img_dir = './check_patches/images/'
save_mask_dir = './check_patches/masks/'
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)

# 🔥 메모리 보호용 상한
#MAX_PATCHES = 10000   # 일단 1만 개까지만 사용 (나중에 늘려도 됨)
# 메모리 터짐 방지

# 학습시킬 폴더 목록
# Kaggle 데이터셋은 train과 val 폴더가 나뉘어 있는데,
# 데이터를 최대한 많이 쓰기 위해 둘 다 가져와서 합칠 예정입니다.
print("데이터 전처리 시작><!!")

# -------------------------------------------------------------------
# 2. 전처리 루프
# -------------------------------------------------------------------
for folder in categories:
    img_files = sorted(glob.glob(base_dir + folder + '/*.tiff'))
    mask_files = sorted(glob.glob(base_dir + folder + '_labels/*.tif'))

    #개수 다른 오류 방지
    if len(img_files) != len(mask_files):
        print(f"⚠ WARNING: {folder} 이미지/마스크 개수 다름")

    for i, (img_f, mask_f) in enumerate(zip(img_files, mask_files)):
        try:
            # 1) 파일 열기
            img = Image.open(img_f).convert('RGB')
            mask = Image.open(mask_f).convert('L')  # 흑백

            # 2) 패딩 (안 나누어떨어지면 검은색 여백 붙이기)
            w, h = img.size
            pad_w = (patch_size - (w % patch_size)) % patch_size
            pad_h = (patch_size - (h % patch_size)) % patch_size

            if pad_w > 0 or pad_h > 0:
                img_padded = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
                mask_padded = Image.new("L", (w + pad_w, h + pad_h), 0)
                img_padded.paste(img, (0, 0))
                mask_padded.paste(mask, (0, 0))
                img, mask = img_padded, mask_padded
                w, h = img.size

                # 3) 타일링 (자르기)
            for y in range(0, h, step):
                for x in range(0, w, step):
                    # 자르기
                    img_patch = img.crop((x, y, x + patch_size, y + patch_size))
                    mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))

                    # =======================================================
                    # [여기!!] 형곤이가 원하던 '눈으로 확인하는 저장' 코드
                    # =======================================================
                    filename = f"{folder}_{i}_{y}_{x}.png"

                    # 1. 항공사진 저장
                    img_patch.save(os.path.join(save_img_dir, filename))

                    # 2. 마스크 저장 (눈에 보이게 255 곱해서 저장)
                    # 그냥 저장하면 너무 까매서 안 보이니까, 시각화용 변수 따로 만듦
                    mask_vis = np.asarray(mask_patch) * 255
                    Image.fromarray(mask_vis.astype('uint8')).save(os.path.join(save_mask_dir, filename))
                    # =======================================================

                    # 데이터 리스트에 추가 (학습용)
                    X.append(np.asarray(img_patch))

                    # 마스크 전처리 (학습용은 0과 1이어야 함)
                    mask_arr = np.asarray(mask_patch) / 255.0
                    mask_arr = (mask_arr > 0.5).astype(int)
                    mask_arr = mask_arr.reshape(patch_size, patch_size, 1)
                    Y.append(mask_arr)

            print(f"{folder} {i}번 이미지 처리 끝... (저장된 패치 누적: {len(X)}개)")

        except Exception as e:
            print(f"에러: {e}")

# -------------------------------------------------------------------
# 3. 마무리 (Numpy 저장)
# -------------------------------------------------------------------
X = np.array(X)
Y = np.array(Y)

print(f"\n총 {len(X)}장의 패치가 생성됐어!")
print("📁 지금 바로 './check_patches' 폴더 열어서 사진 잘 나왔나 확인해봐!")

if len(X) > 0:
    X = X/255
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=777)

    np.save('./binary_X_train.npy', X_train)
    np.save('./binary_X_test.npy', X_test)
    np.save('./binary_Y_train.npy', Y_train)
    np.save('./binary_Y_test.npy', Y_test)
    print("✅ 학습 데이터 저장 완료 (02번 실행 가능)")
else:
    print("❌ 데이터가 0개야. 경로 확인해!")