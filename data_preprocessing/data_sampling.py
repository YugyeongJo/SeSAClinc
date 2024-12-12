import os
import random
import shutil

# 기존 데이터 폴더 경로
base_dir = "C:/develops/SeSAClinc/dataset/object_detecting"
categories = ["done_acne", "done_pigmentation", "done_psoriasis"]

# 새롭게 저장할 경로
output_dir = "C:/develops/SeSAClinc/dataset/object_detecting/result_dataset"
os.makedirs(output_dir, exist_ok=True)

# 랜덤 샘플링 고정을 위해 시드 설정 (재현 가능)
random.seed(42)

# 클래스별 1212개 추출 및 저장
for category in categories:
    # 기존 이미지 및 라벨 폴더 경로
    images_dir = os.path.join(base_dir, category, "images")
    labels_dir = os.path.join(base_dir, category, "labels")
    
    # 새로 저장할 폴더 생성 (done_ 접두사 제거)
    category_name = category.replace("done_", "")  # 'done_' 접두사 제거
    new_images_dir = os.path.join(output_dir, category_name, "images")
    new_labels_dir = os.path.join(output_dir, category_name, "labels")
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)
    
    # 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png"))]
    
    # 1212개 랜덤 추출
    selected_images = random.sample(image_files, 1212)
    
    # 이미지와 라벨 파일 복사
    for image in selected_images:
        # 이미지 파일 복사
        shutil.copy(os.path.join(images_dir, image), new_images_dir)
        
        # 라벨 파일 복사 (확장자 변경)
        label_file = image.rsplit(".", 1)[0] + ".txt"
        shutil.copy(os.path.join(labels_dir, label_file), new_labels_dir)

print("데이터 추출 완료! 새 폴더에 저장되었습니다.")
