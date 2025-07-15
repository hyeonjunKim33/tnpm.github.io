import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# 전처리 함수들
def normalize_zscore(img):
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img) + 1e-8
    norm = (img - mean) / std
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return norm

def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def apply_gaussian_blur(img, ksize=(3, 3)):
    return cv2.GaussianBlur(img, ksize, 0)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def adjust_brightness(img, factor=1.2):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    bright_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def add_gaussian_noise(img, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# 시각화 함수
def show_processed_variants(image_input_folder):
    # 이미지 파일 목록
    image_files = [f for f in os.listdir(image_input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("이미지 폴더가 비어 있습니다.")
        return

    # 무작위로 하나 선택
    selected_file = random.choice(image_files)
    image_path = os.path.join(image_input_folder, selected_file)
    img = cv2.imread(image_path)

    if img is None:
        print(f"[경고] 이미지를 열 수 없음: {image_path}")
        return

    print(f"[INFO] 선택된 파일: {selected_file}")

    # 전처리 수행
    transformations = {
        "Original": img,
        "Z-score Normalized": normalize_zscore(img),
        "CLAHE": apply_clahe(img),
        "Gaussian Blur": apply_gaussian_blur(img),
        "Sharpened": sharpen_image(img),
        "Brightened": adjust_brightness(img),
        "Histogram Eq.": histogram_equalization(img),
        "Gaussian Noise": add_gaussian_noise(img),
    }

    # 결과 시각화
    plt.figure(figsize=(18, 12))
    for i, (title, processed_img) in enumerate(transformations.items(), 1):
        plt.subplot(2, 4, i)
        plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 실행
if __name__ == "__main__":
    image_input_folder = "datasets/brain-tumor/train/images"
    show_processed_variants(image_input_folder)