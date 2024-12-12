import os

def change_file_name(detail_path):
    read_img_path = os.path.join(detail_path, 'images')
    read_label_path = os.path.join(detail_path, 'labels')
    
    # print(read_img_path)
    # print(read_label_path)
    
    image_files = sorted([f for f in os.listdir(read_img_path) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(read_label_path) if f.endswith('.txt')])
    
    # 라벨 파일 이름에서 확장자 제거하여 이미지와 매칭 가능하도록 수정
    label_files_stem = [os.path.splitext(f)[0] for f in label_files]
    
    # 라벨 파일이 없는 이미지 파일 삭제
    for image_file in image_files:
        image_stem = os.path.splitext(image_file)[0]
        if image_stem not in label_files_stem:
            # 라벨이 없는 이미지 파일 삭제
            image_path_to_delete = os.path.join(read_img_path, image_file)
            os.remove(image_path_to_delete)
            print(f"Deleted {image_file} from images folder as it has no corresponding label.")
    
    # 이미지와 라벨 파일 다시 불러오기 (삭제 후 갱신)
    image_files = sorted([f for f in os.listdir(read_img_path) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(read_label_path) if f.endswith('.txt')])
    
    # 이름 변경
    for idx, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        # 새로운 이름 지정
        new_image_name = f"G_{idx+1}.jpg"
        new_label_name = f"G_{idx+1}.txt"
        
        # 파일 경로 지정
        old_image_path = os.path.join(read_img_path, image_file)
        old_label_path = os.path.join(read_label_path, label_file)
        new_image_path = os.path.join(read_img_path, new_image_name)
        new_label_path = os.path.join(read_label_path, new_label_name)
        
        # 파일 이름 변경
        os.rename(old_image_path, new_image_path)
        os.rename(old_label_path, new_label_path)
        
        print(f"Renamed {image_file} to {new_image_name} in images folder")
        print(f"Renamed {label_file} to {new_label_name} in labels folder")
        
if __name__ == '__main__':
    # 모공 데이터셋 저장 위치 & 정리본 저장 위치 
    dataset_path = '../dataset/psoriasis'
    change_file_name(dataset_path)