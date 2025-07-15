import os
import numpy as np
from collections import defaultdict

# 경로 설정
label_folder = 'datasets/brain-tumor/train/labels'      # YOLO 라벨(.txt) 파일 폴더
images_folder = 'datasets/brain-tumor/train/images'     # 이미지 파일 폴더

# 클래스별 수량 저장용
class_counts = defaultdict(int)
unlabeled_count = 0

# 이미지 파일 리스트
image_files = [f for f in os.listdir(images_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 라벨 파일 리스트
label_files_set = set(f for f in os.listdir(label_folder) if f.endswith('.txt'))

for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + '.txt'
    label_path = os.path.join(label_folder, label_file)

    if label_file in label_files_set:
        try:
            data = np.loadtxt(label_path, ndmin=2)  # YOLO format: class x y w h
            classes = data[:, 0].astype(int)
            for cls in classes:
                class_counts[cls] += 1
        except Exception as e:
            print(f"{label_file} 처리 중 오류 발생: {e}")
    else:
        unlabeled_count += 1

# 결과 출력
print(f"{'Class Index':<15}{'Count'}")
print("-" * 30)
for cls, count in sorted(class_counts.items()):
    print(f"{cls:<15}{count}")
print(f"{'No Label Files':<15}{unlabeled_count}")