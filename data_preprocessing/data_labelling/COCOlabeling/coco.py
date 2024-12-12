import os
import json
from PIL import Image

# 폴더 경로 설정
base_dir = "C:/develops/SeSAClinc/ddataset/"
categories = ["acne", "pigmentation", "psoriasis"]
output_json_dir = os.path.join(base_dir, "coco_annotations")
os.makedirs(output_json_dir, exist_ok=True)

# COCO 기본 구조
coco_structure = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "acne"},
        {"id": 2, "name": "pigmentation"},
        {"id": 3, "name": "psoriasis"}
    ]
}

# COCO bbox 변환 함수 (YOLO → COCO)
def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    box_width = width * img_width
    box_height = height * img_height
    return [x_min, y_min, box_width, box_height]

# 고유 ID를 위한 카운터
annotation_id = 1
image_id = 1

# 각 카테고리 처리
for category_id, category in enumerate(categories, start=1):
    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(base_dir, category, split, "images")
        labels_dir = os.path.join(base_dir, category, split, "labels")
        
        # JSON 파일 생성
        json_output_path = os.path.join(output_json_dir, f"{category}_{split}_annotations.json")
        coco_data = {"images": [], "annotations": [], "categories": coco_structure["categories"]}
        
        # 이미지 파일 순회
        for image_file in os.listdir(images_dir):
            if not image_file.endswith((".jpg", ".png")):
                continue
            
            # 이미지 정보 추가
            img_path = os.path.join(images_dir, image_file)
            img = Image.open(img_path)
            img_width, img_height = img.size
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": img_width,
                "height": img_height
            })
            
            # 라벨 파일 읽기
            label_file = os.path.join(labels_dir, image_file.rsplit(".", 1)[0] + ".txt")
            if not os.path.exists(label_file):
                continue
            
            with open(label_file, "r") as f:
                for line in f:
                    # YOLO 형식 값 읽기
                    values = line.split()
                    if len(values) < 5:
                        continue  # 남은 값이 4의 배수가 아니라면 건너뛰기
                    
                    class_id, x_center, y_center, width, height = map(float, values[:5])
                    
                    # COCO 형식 변환
                    bbox = yolo_to_coco_bbox([x_center, y_center, width, height], img_width, img_height)
                    area = bbox[2] * bbox[3]
                    
                    # COCO annotation 추가
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id) + 1,  # YOLO class_id는 0부터 시작하므로 +1
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += 1
        
        # COCO JSON 저장
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(coco_data, json_file, indent=4)
        
        print(f"{category} - {split}: JSON 파일 생성 완료! -> {json_output_path}")

print("모든 COCO 라벨링 JSON 파일 생성 완료!")
