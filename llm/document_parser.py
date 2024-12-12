import os
import requests
from bs4 import BeautifulSoup
from typing import List
import pandas as pd
import pdfplumber
import easyocr

# 텍스트를 단어 단위로 분리하는 함수
def tokenize_txt(text):
    return text.split()

# TXT 파일에서 텍스트 추출
def parse_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines() 
            words = []
            for line in lines:
                temp = tokenize_txt(line.strip()) + ['<EOS>']
                words.extend(temp)
            return words
    except Exception as e:
        print(f"TXT 파싱 중 오류: {e}")
        return []
    
# PDF 파일에서 텍스트 추출 
def parse_pdf(file_path):
    words = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    temp = tokenize_txt(text) + ['<EOS>']
                    words.extend(temp)
        return words
    except Exception as e:
        print(f"PDF 파싱 중 오류: {e}")
        return []
    
# 웹 페이지에서 텍스트 추출
def parse_web(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find_all(["p", "h1", "h2", "h3"])
        words = []
        for tag in content:
            temp = tokenize_txt(tag.get_text(separator=" ", strip=True)) + ['<EOS>']
            words.extend(temp)
        return words
    except Exception as e:
        print(f"웹 파싱 중 오류: {e}")
        return []

# Excel 에서 텍스트 추출
def parse_excel(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 모든 시트를 읽어옴
        sheets = pd.read_excel(file_path, sheet_name=None)
        words = []
        
        for sheet_name, df in sheets.items():
            print(f"처리 중: {sheet_name} 시트")
            for cell in df.astype(str).stack().tolist():
                temp = tokenize_txt(cell) + ['<EOS>'] 
                words.extend(temp)
        
        return words
    except Exception as e:
        print(f"Excel 파싱 중 오류: {e}")
        return []
    
# EasyOCR로 이미지에서 텍스트 추출
def parse_image(image_path):
    try:
        # EasyOCR Reader 초기화
        reader = easyocr.Reader(['en', 'ko'])  # 영어와 한국어 지원
        results = reader.readtext(image_path, detail=0)  # detail=0: 텍스트만 반환
        
        return results  # 텍스트 리스트를 직접 반환
    except Exception as e:
        print(f"EasyOCR 처리 중 오류: {e}")
        return []

def save_parse(words, output_path):
    try:
        # NaN 값 제거 후 파일 저장
        words = [word for word in words if word.lower() != 'nan']
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(' '.join(words))
        print(f"파일 저장 완료 : {output_path}")
    except Exception as e:
        print(f"파일 저장 중 오류: {e}")

if __name__ == '__main__':
    # Excel 파일에서 "화장품" 시트를 읽어옴
    file_path = "./dataset_list.xlsx"
    df = pd.read_excel(file_path, sheet_name="document", header=0)

    print(df)
    
    # "형식"을 key, "경로"를 value로 하는 딕셔너리 생성
    format_to_path = df.groupby("유형")["경로"].apply(list).to_dict()

    print(format_to_path)
    
    # "형식"별로 경로에 맞는 텍스트 추출 후 저장
    for file_type, paths in format_to_path.items():
        for path in paths:
            all_words = []

            # 각 경로에 대해 텍스트 추출
            if file_type == "web":
                words = parse_web(path)
            elif file_type == "pdf":
                words = parse_pdf(path)
            elif file_type == "txt":
                words = parse_txt(path)
            elif file_type == "excel":
                words = parse_excel(path)
            elif file_type == "img":
                words = parse_image(path)
            else:
                print(f"지원되지 않는 형식: {file_type}")
                continue

            all_words.extend(words)

            # 경로에 맞는 '데이터셋명' 추출
            dataset_name = df.loc[df["경로"] == path, "데이터셋명"].values[0]  # 해당 경로에 해당하는 '데이터셋명' 추출

            # output_file 디렉토리가 없으면 생성
            output_dir = "./output_file"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 각 경로에 대해 별도의 파일로 저장
            output_file_path = os.path.join(output_dir, f"{dataset_name}.txt")

            # 파싱 결과 저장
            save_parse(all_words, output_file_path)
