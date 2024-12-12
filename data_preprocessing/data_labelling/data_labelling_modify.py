import os

# 경로 설정
input_folder = r'C:\develops\SeSAClinc\dataset\done_pores\labels'  # 기존 YOLO 라벨 파일들이 있는 폴더
output_folder = r'C:\develops\SeSAClinc\dataset\done_pores\modified'  # 수정된 파일 저장 폴더

# output_folder가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"'{output_folder}' 폴더를 생성했습니다.")

# labels 폴더 내 모든 .txt 파일 처리
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):  # .txt 파일만 처리
        input_file = os.path.join(input_folder, file_name)  # 원본 파일 경로
        output_file = os.path.join(output_folder, file_name)  # 저장 파일 경로

        # 동일한 파일이 output_folder에 이미 존재하면 건너뛰기
        if os.path.exists(output_file):
            print(f"'{output_file}' 파일이 이미 존재합니다. 건너뜁니다.")
            continue

        # 파일 읽고 클래스 인덱스 수정
        with open(input_file, 'r') as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            parts = line.strip().split()  # 공백 기준으로 분리
            parts[0] = '4'  # 클래스 인덱스를 0으로 변경 (필요에 따라 수정 가능)
            modified_lines.append(' '.join(parts))  # 수정된 줄 추가

        # 수정된 내용을 새로운 파일에 저장
        with open(output_file, 'w') as file:
            file.write('\n'.join(modified_lines))

        print(f"수정된 파일이 '{output_file}'에 저장되었습니다.")

print("모든 파일 처리가 완료되었습니다.")
