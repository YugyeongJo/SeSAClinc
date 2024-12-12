import os
import random
import shutil

# 기존 데이터 경로
base_dir = "C:/develops/SeSAClinc/dataset/object_detecting/result_dataset"
categories = ["acne", "pigmentation", "psoriasis"]

# 랜덤 시드 설정
random.seed(42)

# 데이터셋 나누기
split_ratios = {
    "train": 0.7,
    "valid": 0.2,
    "test": 0.1
}

# 데이터셋 나누기 함수
def split_and_save_data(category_dir, category_name):
    # 원본 이미지 및 라벨 폴더 경로
    images_dir = os.path.join(category_dir, "images")
    labels_dir = os.path.join(category_dir, "labels")
    
    # train, valid, test 폴더 생성
    for split in ["train", "valid", "test"]:
        split_images_dir = os.path.join(category_dir, split, "images")
        split_labels_dir = os.path.join(category_dir, split, "labels")
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
    
    # 전체 이미지 리스트 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(image_files)  # 데이터를 랜덤하게 섞음

    # 각 비율에 따라 데이터 나누기
    total_images = len(image_files)
    train_end = int(total_images * split_ratios["train"])
    valid_end = train_end + int(total_images * split_ratios["valid"])

    split_data = {
        "train": image_files[:train_end],
        "valid": image_files[train_end:valid_end],
        "test": image_files[valid_end:]
    }
    
    # 각 데이터셋에 따라 파일 복사
    for split, files in split_data.items():
        for image in files:
            # 이미지 파일 복사
            src_image_path = os.path.join(images_dir, image)
            dest_image_path = os.path.join(category_dir, split, "images", image)
            shutil.copy(src_image_path, dest_image_path)
            
            # 라벨 파일 복사
            label_file = image.rsplit(".", 1)[0] + ".txt"
            src_label_path = os.path.join(labels_dir, label_file)
            dest_label_path = os.path.join(category_dir, split, "labels", label_file)
            shutil.copy(src_label_path, dest_label_path)

# 카테고리별로 데이터 나누기
for category in categories:
    category_dir = os.path.join(base_dir, category)
    split_and_save_data(category_dir, category)

print("데이터셋 분리 완료! 각 폴더에 train, valid, test 데이터가 저장되었습니다.")
