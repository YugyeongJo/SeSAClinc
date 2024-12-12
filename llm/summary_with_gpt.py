import os
import json
import openai
from dotenv import load_dotenv

# .env 파일의 내용을 로드합니다.
load_dotenv()

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

llm = 'gpt-4o-mini'

# OpenAI API 요청 함수
def get_advice(skin_data, model=llm):
    """
    사용자의 피부 상태를 기반으로 OpenAI를 통해 피부 관리 요약문을 생성.
    """
    prompt = f"""
    사용자의 피부 상태에 따라 적합한 피부 관리 방법과 피해야 할 성분에 대해 간단히 설명해주세요. 설명은 3줄 이내로 요약하세요.
    사용자의 피부 상태는 JSON 형식으로 주어집니다.
    출력은 한국어로만 제공하세요.
    
    예시 입력:
    {{
        "probabilities": {{
            "flushing": 100.0,
            "normal": 0.0
        }}
    }}
    
    사용자 입력:
    {json.dumps(skin_data, indent=4)}
    
    출력 형식:
    {{
        "advice": "string"
    }}
    """
    try:
        # GPT 모델 호출
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant specializing in skincare."},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        # GPT 응답 가져오기
        raw_response = response.choices[0].message.content
        
        # JSON으로 변환
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed_response = {"error": "응답이 JSON 형식이 아닙니다.", "raw_response": raw_response}
        
        return parsed_response
    except Exception as e:
        return {"error": str(e)}

# 사용자의 피부 상태 데이터 (예시 JSON)
user_skin_data = {
    "probabilities": {
        "flushing": 100.0,
        "normal": 0.0
    }
}

# 피부 관리 요약 요청
result = get_advice(user_skin_data)

# 결과 출력
print("결과:", result)
