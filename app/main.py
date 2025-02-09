from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat import chat_stream
import json
from fastapi.responses import StreamingResponse

app = FastAPI(title="스마트스토어 FAQ 챗봇")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # 명시적인 오리진 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/chat")
async def chat_endpoint(
    message: str = Query(...),
    session_id: str = Query('default-session')
):
    # ChatRequest 생성
    request = ChatRequest(message=message, session_id=session_id)

    async def event_generator():
        try:
            async for response in chat_stream(request):
                # 각 응답에 대해 이벤트 데이터 생성
                event_data = json.dumps({
                    "event": "message",
                    "data": json.dumps({
                        "answer": response.answer,
                        "follow_up_questions": response.follow_up_questions,
                        "sources": response.sources,
                        "confidence_score": response.confidence_score
                    })
                })
                yield f"data: {event_data}\n\n"

            # 스트림 종료 이벤트
            yield f"data: {json.dumps({'event': 'end', 'data': ''})}\n\n"

        except Exception as e:
            error_data = json.dumps({
                "event": "error",
                "data": str(e)
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)