import os
import torch
import numpy as np
import faiss  # FAISS 라이브러리
from transformers import ElectraTokenizer, ElectraModel
from typing import List, Tuple
from kiwipiepy import Kiwi

# KoELECTRA 모델 및 토크나이저 로드
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
kiwi = Kiwi()

# EOS 기준으로 텍스트를 문장 단위로 분리하는 함수
def split_text_by_eos(text: str) -> List[str]:
    return [sentence.strip() for sentence in text.split("<EOS>") if sentence.strip()]

# 청크 사이즈를 문장 길이에 맞게 동적으로 결정하는 함수
def determine_chunk_size(sentence: str, base_size: int = 150, max_size: int = 500) -> int:
    length = len(sentence)
    if length >= max_size:
        return max_size
    elif length <= base_size:
        return base_size
    else:
        return int(base_size + (length - base_size) * 0.5)

# KoELECTRA 임베딩 생성 함수 (임베딩과 문장 반환)
def generate_electra_embeddings_with_text(texts: List[str]) -> List[Tuple[np.ndarray, str]]:
    embeddings_with_texts = []
    for text in texts:
        sentences = split_text_by_eos(text)

        for sentence in sentences:
            # 각 문장에 대해 동적으로 청크 사이즈 계산
            dynamic_chunk_size = determine_chunk_size(sentence)
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

            # 임베딩과 원본 문장 함께 저장
            embeddings_with_texts.append((embedding, sentence))

    return embeddings_with_texts

# TXT 파일에서 텍스트 읽기
def extract_text_from_txt(file_path: str) -> str:
    print(f"TXT 파일 파싱 중: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"TXT 파일 파싱 중 오류 발생: {e}")
        return ""

# 디렉토리 내의 모든 텍스트 파일에서 KoELECTRA 임베딩 생성
def generate_embeddings_from_files(file_paths: List[str]) -> List[Tuple[np.ndarray, str]]:
    all_embeddings = []

    # 각 파일에서 텍스트를 읽고 임베딩 생성
    for file_path in file_paths:
        print(f"파일 {file_path} 처리 중...")
        all_text = extract_text_from_txt(file_path)

        if all_text:
            embeddings = generate_electra_embeddings_with_text([all_text])
            all_embeddings.extend(embeddings)

    return all_embeddings

# FAISS 벡터 DB 생성
def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    embedding_dim = embeddings[0].shape[0]  # 임베딩 차원 (KoELECTRA: 768)
    index = faiss.IndexFlatL2(embedding_dim)  # L2 거리 기반 Index 생성
    index.add(np.array(embeddings))  # 임베딩 추가
    print(f"FAISS Index에 {index.ntotal}개 벡터 추가 완료.")
    return index

# FAISS Index 저장
def save_faiss_index(index: faiss.IndexFlatL2, file_path: str):
    faiss.write_index(index, file_path)
    print(f"FAISS Index가 {file_path}에 저장되었습니다.")

# FAISS Index 로드
def load_faiss_index(file_path: str) -> faiss.IndexFlatL2:
    index = faiss.read_index(file_path)
    print(f"FAISS Index가 {file_path}에서 로드되었습니다.")
    return index

# 임베딩과 문장을 텍스트 파일로 저장
def save_embeddings_to_txt(embeddings_with_texts: List[Tuple[np.ndarray, str]], output_file_path: str):
    with open(output_file_path, "w", encoding="utf-8") as file:
        for embedding, sentence in embeddings_with_texts:
            # 임베딩 벡터를 문자열로 변환하여 저장
            embedding_str = " ".join(map(str, embedding))
            # 문장과 임베딩을 탭으로 구분해서 저장
            file.write(f"{embedding_str}\t{sentence}\n")
    print(f"임베딩과 문장이 {output_file_path}에 저장되었습니다.")

if __name__ == "__main__":
    # 텍스트 파일들이 있는 디렉토리 경로
    directory_path = "./output_file"  # 텍스트 파일 경로를 설정하세요.

    # 디렉토리 내의 모든 .txt 파일 경로 가져오기
    txt_file_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".txt")
    ]

    print(txt_file_paths)

    # 파일 리스트에서 임베딩 생성
    embeddings_with_texts = generate_embeddings_from_files(txt_file_paths)
    
    # 임베딩과 문장을 텍스트 파일로 저장
    save_embeddings_to_txt(embeddings_with_texts, './koelectra_embeddings.txt')

    if embeddings_with_texts:
        print("임베딩 생성 완료!")
        print(f"생성된 임베딩 개수: {len(embeddings_with_texts)}")

        # 임베딩만 추출
        embeddings = [embedding for embedding, _ in embeddings_with_texts]

        # FAISS Index 생성
        faiss_index = create_faiss_index(embeddings)

        # FAISS Index 저장
        save_faiss_index(faiss_index, "./faiss_index_file.index")
    else:
        print("임베딩 생성 실패")
