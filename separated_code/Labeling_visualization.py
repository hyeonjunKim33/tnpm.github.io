import os
import random
import cv2
import matplotlib.pyplot as plt

# 경로 (본인 환경에 맞게 변경)
image_folder = r"dataset/brain-tumor/train/images"
label_folder = r"dataset/brain-tumor/train/labels"

# 이미지 리스트
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

sample_count = 50
if len(image_files) < sample_count:
    sample_count = len(image_files)

sample_files = random.sample(image_files, sample_count)

cols, rows = 10, 5
fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, ax in enumerate(axes.flat):
    if i >= sample_count:
        ax.axis('off')
        continue

    img_name = sample_files[i]
    img_path = os.path.join(image_folder, img_name)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(label_folder, label_name)

    # 이미지 읽기 (OpenCV는 BGR)
    img = cv2.imread(img_path)
    if img is None:
        ax.axis('off')
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # 라벨 읽어서 박스 그리기
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, bw, bh = parts
                if cls != '1':  # positive class만 박스 그림
                    continue
                x_c, y_c, bw, bh = map(float, [x_c, y_c, bw, bh])

                # YOLO 형식 중심좌표->절대좌표 변환
                xmin = int((x_c - bw / 2) * w)
                ymin = int((y_c - bh / 2) * h)
                xmax = int((x_c + bw / 2) * w)
                ymax = int((y_c + bh / 2) * h)

                # 박스 그리기 (빨간색, 두께 2)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # 이미지 크기 조정 (가로 90픽셀)
    base_width = 90
    scale_ratio = base_width / w
    new_h = int(h * scale_ratio)
    img = cv2.resize(img, (base_width, new_h))

    ax.imshow(img)
    ax.set_title(img_name, fontsize=5)
    ax.axis('off')

plt.tight_layout()
plt.show()