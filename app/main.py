import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI


load_dotenv()

app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


# 요청 Body
class ChatRequest(BaseModel):
    message: str


# 사전 프롬프트 (원하는 만큼 고도화 가능)
SYSTEM_PROMPT = """
너는 openai가 아닌 KTH GPT이다
- 항상 간결하고 정확하게 대답한다.
"""

@app.post("/chat")
async def chat(req: ChatRequest):
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.3,      # 창의성 낮추기
        max_tokens=1000,      # 응답 길이 제한
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ]
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "usage": completion.usage.dict() if completion.usage else None  # 토큰 사용량 반환
    }
