import os
import pandas as pd

# 경로 설정
base_path = r"C:/develops/SeSAClinc/ddataset/"
categories = ["acne", "pigmentation", "psoriasis"]
splits = ["train", "test", "valid"]
empty_files = []  # 비어있는 파일을 저장할 리스트

# 탐색 시작
for category in categories:
    for split in splits:
        labels_path = os.path.join(base_path, category, split, "labels")  # labels 디렉터리 경로
        if not os.path.exists(labels_path):
            print(f"경로가 존재하지 않습니다: {labels_path}")
            continue

        for file_name in os.listdir(labels_path):
            if file_name.endswith(".txt"):  # .txt 파일만 확인
                file_path = os.path.join(labels_path, file_name)
                # 파일이 비어있는지 확인
                df = pd.read_csv(file_path, header=None)
                if len(df) <= 0:
                    empty_files.append(file_path)
                
# 결과 출력
if empty_files:
    print("비어있는 파일 목록:")
    for file in empty_files:
        print(file)
else:
    print("비어있는 파일이 없습니다.")
