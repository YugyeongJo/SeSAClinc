import os
import numpy as np

# 폴더 경로 설정
label_folder = r"C:\develops\SeSAClinc\dataset\object_detecting\done_psoriasis\labels"

# 이미지 크기 (YOLO 모델의 경우 실제 이미지 크기로 정규화 필요)
# 데이터가 정규화된 상태라면 이미지 크기를 1로 설정
image_width = 1
image_height = 1

def polygon_to_yolo_format(polygon_line, image_width, image_height):
    """폴리곤 데이터를 YOLO 형식으로 변환"""
    # 라벨 데이터 파싱
    data = list(map(float, polygon_line.split()))
    class_index = int(data[0])  # 클래스 인덱스
    points = np.array(data[1:]).reshape(-1, 2)  # 좌표 (x, y)로 변환
    
    # Bounding Box 계산
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    # 정규화
    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    return f"{class_index} {center_x} {center_y} {width} {height}"

def process_label_files(label_folder, image_width, image_height):
    """폴더 내의 모든 txt 파일을 YOLO 형식으로 변환"""
    for file_name in os.listdir(label_folder):
        if file_name.endswith(".txt"):  # txt 파일만 처리
            file_path = os.path.join(label_folder, file_name)
            
            # 파일 읽기
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            # 변환된 데이터를 저장할 리스트
            yolo_data = []
            for line in lines:
                if line.strip():  # 빈 줄은 건너뜀
                    yolo_format = polygon_to_yolo_format(line, image_width, image_height)
                    yolo_data.append(yolo_format)
            
            # 변환된 데이터를 동일한 파일에 저장
            with open(file_path, "w") as file:
                file.write("\n".join(yolo_data))

# 실행
process_label_files(label_folder, image_width, image_height)
print("YOLO 포맷 변환이 완료되었습니다!")
