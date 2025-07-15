import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
# 경로 설정
image_folder = 'dataset/brain-tumor/train/images'
label_folder = 'dataset/brain-tumor/train/labels'

# 클래스별 이미지 매핑 딕셔너리
class_to_images = {}

# 모든 라벨 파일 순회
label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
for label_file in label_files:
    label_path = os.path.join(label_folder, label_file)
    base_name = os.path.splitext(label_file)[0]
    
    # 이미지 확장자 대응 (.jpg, .png, etc.)
    possible_extensions = ['.jpg', '.png', '.jpeg']
    image_path = None
    for ext in possible_extensions:
        temp_path = os.path.join(image_folder, base_name + ext)
        if os.path.exists(temp_path):
            image_path = temp_path
            break
    if image_path is None:
        continue

    try:
        labels = np.loadtxt(label_path, ndmin=2)
        class_ids = labels[:, 0].astype(int)
        for cls in np.unique(class_ids):
            class_to_images.setdefault(cls, []).append(image_path)
    except Exception as e:
        print(f"[오류] {label_file} 처리 실패: {e}")

# ✅ 클래스 이름 매핑
class_titles = {
    0: "Brain-Tumor: NO (class 0)",
    1: "Brain-Tumor: YES (class 1)"
}

# ✅ 시각화 함수
def show_class_images_grid(class_index, image_paths, max_images=30, cols=10, rows=3):
    sampled = random.sample(image_paths, min(len(image_paths), max_images))

    plt.figure(figsize=(cols * 2, rows * 2))
    for i, img_path in enumerate(sampled):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(os.path.basename(img_path), fontsize=8)
        plt.axis('off')

    title = class_titles.get(class_index, f"Class {class_index}")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

# ✅ 모든 클래스에 대해 시각화 실행
for cls, img_list in sorted(class_to_images.items()):
    if len(img_list) < 1:
        continue
    show_class_images_grid(cls, img_list, max_images=30, cols=10, rows=3)
