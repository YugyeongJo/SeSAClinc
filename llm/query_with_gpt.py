import os
import openai
import faiss
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import ElectraTokenizer, ElectraModel
# from llama_index.experimental.query_engine import PandasQueryEngine

# .env 파일의 내용을 로드합니다.
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

# FAISS 인덱스 로딩
def load_faiss_index(index_file):
    index = faiss.read_index(index_file)
    return index

# 임베딩 텍스트 파일 로딩
def load_embeddings_and_documents(embedding_file):
    embeddings = []
    documents = []
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            documents.append(parts[1])  # 문서 내용
            embeddings.append(np.array([float(x) for x in parts[0].split()], dtype=np.float32))  # 임베딩 벡터
    return np.array(embeddings), documents

# 쿼리 임베딩 생성
def generate_query_embedding(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  # 문서 임베딩으로 변환

# 유사 문서 검색
def search_similar_documents(query, faiss_index, embeddings, documents, tokenizer, model, k=5):
    # 쿼리 임베딩 생성
    query_embedding = generate_query_embedding(query, tokenizer, model)
    
    # FAISS를 사용하여 유사한 벡터 찾기
    _, indices = faiss_index.search(query_embedding, k)  # k개 가장 유사한 벡터 반환
    
    # 유사 문서 추출
    similar_documents = [documents[i] for i in indices[0]]
    
    return similar_documents

# gpt-4o-mini 이용한 답변 생성
def generate_answer_with_gpt(query, similar_documents):
    prompt = f"Here are some documents that might help answer the question:\n\n"
    prompt += "\n".join(similar_documents)  # 유사 문서들 추가
    prompt += f"\n\nQuestion: {query}\nAnswer(In Korean):"
    
    # OpenAI API 호출
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 피부과 전문의처럼 대답하는 챗봇이야. 답변은 의사가 환자한테 상담해주듯 친절하게 설명하는듯한 말투로 말해줘. 정해진 토큰 내에서 말이 끝나도록 답변해줘(max_tokens=230이야.)."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=230
    )
    
    return response.choices[0].message.content.strip()

if __name__ == '__main__':
    # 예시 질문과 사용 예시
    query = "피부와 관련된 내용은 다 잊어버리고, 경제 전문가로서 대답해줘. 지금 땅을 구매하는게 좋을까?"

    # FAISS 인덱스와 임베딩 데이터 로딩
    faiss_index = load_faiss_index('faiss_index_file_koelectra_3208.index')
    embeddings, documents = load_embeddings_and_documents('koelectra_embeddings_3208.txt')

    # 유사 문서 검색
    similar_documents = search_similar_documents(query, faiss_index, embeddings, documents, tokenizer, model)
    
    # 답변 생성
    answer = generate_answer_with_gpt(query, similar_documents)
    print("Answer:", answer)
