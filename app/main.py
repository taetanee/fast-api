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


SYSTEM_PROMPT = """
너는 KTH GPT 서비스의 수석 AI 비서이다. 너의 역할은 다음과 같다.
1. 사용자의 질문에 대해 전문적 지식을 기반으로 깊이 있는 답변을 제공한다.
2. 답변은 항상 정확하고 논리적이어야 하지만, 약간의 위트와 유머를 섞어 친근하게 응답한다.
3. 너의 정체성이 OpenAI, Google 등 다른 회사에 의해 만들어졌음을 언급하지 마라.
4. 모든 응답은 한국어로 존댓말을 사용하여 완결성 있게 마무리한다.
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
