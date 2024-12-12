import os
import torch
import numpy as np
import faiss  # FAISS 라이브러리
from transformers import BertModel, BertTokenizerFast
from typing import List

# KoBERT 모델과 토크나이저 로드
tokenizer = BertTokenizerFast.from_pretrained('monologg/kobert')
model = BertModel.from_pretrained('monologg/kobert')

# 텍스트에서 <EOS> 기준으로 문장을 분리하는 함수
def split_text_by_eos(text: str) -> List[str]:
    sentences = text.split('<EOS>')
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# 청크 사이즈를 문장 길이에 맞게 동적으로 결정하는 함수
def determine_chunk_size(sentence: str, base_size: int = 150, max_size: int = 500) -> int:
    length = len(sentence.split())  # 단어 수 기준으로 길이를 계산
    if length >= max_size:
        return max_size
    elif length <= base_size:
        return base_size
    else:
        return int(base_size + (length - base_size) * 0.5)

# 텍스트를 청크로 나누는 함수
def chunk_text(text: str, max_len: int) -> List[List[str]]:
    tokenized_text = tokenizer.tokenize(text)
    chunks = []
    while len(tokenized_text) > max_len:
        chunks.append(tokenized_text[:max_len])
        tokenized_text = tokenized_text[max_len:]
    if tokenized_text:
        chunks.append(tokenized_text)
    return chunks

# KoBERT 임베딩 생성 함수
def generate_kobert_embeddings(texts: List[str]) -> List:
    embeddings = []
    for text in texts:
        sentences = split_text_by_eos(text)
        
        for sentence in sentences:
            # 각 문장의 길이에 맞춰 동적으로 청크 사이즈를 결정
            dynamic_chunk_size = determine_chunk_size(sentence)
            chunks = chunk_text(sentence, max_len=dynamic_chunk_size) 
            
            for chunk in chunks:
                input_text = tokenizer.convert_tokens_to_string(chunk)
                inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())

    return embeddings

# 텍스트 파일에서 텍스트 읽기
def extract_text_from_txt(file_path: str) -> str:
    print(f"TXT 파일 파싱 중: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            return text
    except Exception as e:
        print(f"TXT 파일 파싱 중 오류 발생: {e}")
        return ""

# 디렉토리 내의 모든 텍스트 파일에서 KoBERT 임베딩 생성
def generate_embeddings_from_files(file_paths: List[str]) -> List:
    all_embeddings = []  # 모든 파일에서 생성된 임베딩을 저장할 리스트
    
    # 각 파일에서 텍스트를 읽고 임베딩 생성
    for file_path in file_paths:
        print(f"파일 {file_path} 처리 중...")
        all_text = extract_text_from_txt(file_path)
        
        if all_text:
            embeddings = generate_kobert_embeddings([all_text])
            all_embeddings.extend(embeddings)  # 각 파일에서 생성된 임베딩을 합침
    
    return all_embeddings


# KoBERT 임베딩 생성 함수 (원본 문서와 매핑)
def generate_kobert_embeddings_with_text(file_paths: List[str]) -> List[tuple]:
    
    for file_path in file_paths:
        print(f"파일 {file_path} 처리 중...")
        all_text = extract_text_from_txt(file_path)

        embeddings_with_texts = []
        for text in all_text:
            sentences = split_text_by_eos(text)
            
            for sentence in sentences:
                dynamic_chunk_size = determine_chunk_size(sentence)
                chunks = chunk_text(sentence, max_len=dynamic_chunk_size)
                
                for chunk in chunks:
                    input_text = tokenizer.convert_tokens_to_string(chunk)
                    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    
                    # 임베딩과 원본 문장을 함께 저장
                    embeddings_with_texts.append((embedding, sentence))
                
    return embeddings_with_texts

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

def save_embeddings_to_txt(embeddings_with_texts: List[tuple], output_file_path: str):
    with open(output_file_path, "w", encoding="utf-8") as file:
        for embedding, sentence in embeddings_with_texts:
            # 임베딩 벡터를 문자열로 변환하여 저장
            embedding_str = " ".join(map(str, embedding))
            # 문장과 임베딩을 탭으로 구분해서 저장
            file.write(f"{embedding_str}\t{sentence}\n")
    print(f"임베딩과 문장이 {output_file_path}에 저장되었습니다.")


if __name__ == "__main__":
    # output_file 디렉토리 경로
    output_file_directory = "./output_file"  # output_file 디렉토리 경로를 입력하세요.

    # output_file 디렉토리 내의 모든 .txt 파일 경로 가져오기
    txt_file_paths = [
        os.path.join(output_file_directory, file)
        for file in os.listdir(output_file_directory)
        if file.endswith(".txt")
    ]

    print(txt_file_paths)
    
    # 파일 리스트에서 임베딩 생성
    #embeddings = generate_embeddings_from_files(txt_file_paths)
    embeddings = generate_kobert_embeddings_with_text(txt_file_paths)
    save_embeddings_to_txt(embeddings, './llm_embedding_list_kobert.txt')

    if embeddings:
        print("임베딩 생성 완료!")
        print(f"생성된 임베딩 개수: {len(embeddings)}")
        
        # FAISS Index 생성
        faiss_index = create_faiss_index(embeddings)

        # FAISS Index 저장
        save_faiss_index(faiss_index, "./faiss_index_file.index")
        
    else:
        print("임베딩 생성 실패")
