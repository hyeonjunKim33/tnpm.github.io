import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 폴더 경로 
image_folder = r"dataset/brain-tumor/train/images"

# 이미지 파일 리스트
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

sample_count = 50
if len(image_files) < sample_count:
    sample_count = len(image_files)

sample_files = random.sample(image_files, sample_count)

cols, rows = 10, 5
fig, axes = plt.subplots(rows, cols, figsize=(15, 7))  # 전체 크기 줄임
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, ax in enumerate(axes.flat):
    if i < sample_count:
        img_path = os.path.join(image_folder, sample_files[i])
        img = Image.open(img_path)
        img = img.convert("RGB")

        base_width = 90
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))

        try:
            resample_mode = Image.Resampling.LANCZOS
        except AttributeError:
            resample_mode = Image.ANTIALIAS

        img = img.resize((base_width, h_size), resample_mode)

        ax.imshow(img)
        ax.set_title(sample_files[i], fontsize=5)
    ax.axis('off')