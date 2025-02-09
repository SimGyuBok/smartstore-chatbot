from openai import OpenAI
from typing import List, Dict, AsyncGenerator, Optional
import json
import asyncio
from datetime import datetime

from app.config import get_settings
from app.database.vector_store import VectorStore
from app.models.chat import ChatMessage, ChatResponse, ChatRequest, ChatHistory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)
vector_store = VectorStore()

# 대화 히스토리 관리
chat_histories: Dict[str, ChatHistory] = {}

def get_chat_history(session_id: str) -> ChatHistory:
    return chat_histories.get(session_id, ChatHistory(messages=[], session_id=session_id))

def update_chat_history(session_id: str, message: ChatMessage):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatHistory(messages=[], session_id=session_id)
    
    chat_histories[session_id].messages.append(message)
    
    if len(chat_histories[session_id].messages) > 20:
        chat_histories[session_id].messages = chat_histories[session_id].messages[-20:]

def is_smartstore_related(query: str) -> str:
    system_prompt = """
    다음 질문이 네이버 스마트스토어와 직접적으로 관련이 없다면, 
    해당 질문에서 스마트스토어와 연결지을 수 있는 키워드나 주제를 제안해주세요.
    
    형식:
    - 관련 없음: "no"
    - 간접적 연관성: "related: [스마트스토어와 연결 가능한 제안]"
    
    질문: {query}
    """.format(query=query)
    
    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()


def calculate_confidence(similar_faqs: Dict) -> float:
    try:
        distances = similar_faqs.get('distances', [[0]])[0]
        return max(1 - min(distances), 0)
    except Exception as e:
        logger.error(f"Confidence calculation error: {e}")
        return 0.0

def generate_follow_up_questions(query: str, response: str) -> List[str]:
    try:
        # 명시적으로 더 많은 FAQ 결과 요청
        similar_faqs = vector_store.query_similar(query, n_results=10)
        related_questions = [meta['question'] for meta in similar_faqs['metadatas'][0]]
        
        prompt = f"""
        스마트스토어 관련 FAQ를 기반으로 사용자의 질문에 대해 더 깊이 알아볼 수 있는 후속 질문을 2-3개 생성해주세요.
        질문은 "~을/를 알려드릴까요?" 형식으로 작성하세요.

        원래 질문: {query}
        답변: {response}
        참고할 FAQ 질문들:
        {json.dumps(related_questions, ensure_ascii=False)}
        """
        
        follow_up_response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": "너는 스마트스토어 FAQ 챗봇의 친절한 상담원이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        
        questions = follow_up_response.choices[0].message.content.strip().split('\n')
        
        # 질문 형식 보장 및 필터링
        formatted_questions = [
            q.strip() for q in questions 
            if q.strip() and (
                q.endswith('알려드릴까요?') or 
                q.endswith('필요하신가요?') or 
                q.endswith('궁금하신가요?')
            )
        ]
        
        return formatted_questions[:3] or [
            "스마트스토어에 대해 더 알고 싶으신가요?",
            "추가 정보가 필요하신가요?",
            "다른 궁금한 점이 있으신가요?"
        ]
        
    except Exception as e:
        logger.error(f"Follow-up question generation error: {e}")
        return [
            "스마트스토어에 대해 더 알고 싶으신가요?",
            "추가 정보가 필요하신가요?",
            "다른 궁금한 점이 있으신가요?"
        ]

async def chat_stream(request: ChatRequest) -> AsyncGenerator[ChatResponse, None]:
    #logger.info(f"Starting chat stream for query: {request.message}")
    try:
        # 스마트스토어 관련성 확인
        is_related = is_smartstore_related(request.message)
        
        # 스마트스토어와 직접적으로 관련 없는 경우 처리
        if is_related.startswith("no") or is_related.startswith("related:"):
            response = ChatResponse(
                answer="저는 스마트스토어 FAQ를 위한 챗봇입니다. 스마트스토어에 대한 질문을 부탁드립니다.",
                confidence_score=0.1
            )
            
            try:
                follow_up_prompt = f"""
                사용자의 질문은 스마트스토어와 직접적인 관련이 없습니다. 
                하지만 다음 질문에서 스마트스토어와 연결 가능한 창의적인 후속 질문을 2-3개 제안해주세요.
                
                원래 질문: {request.message}
                
                후속 질문 형식: "~ 알려드릴까요?"
                """
                
                follow_up_response = client.chat.completions.create(
                    model=settings.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "스마트스토어와 관련된 창의적인 연결 질문을 제안하세요."},
                        {"role": "user", "content": follow_up_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                
                questions = follow_up_response.choices[0].message.content.strip().split('\n')
                
                # 질문 형식 보장
                response.follow_up_questions = [
                    q.strip() for q in questions 
                    if q.strip() and (
                        q.endswith('알려드릴까요?') or 
                        q.endswith('궁금하신가요?')
                    )
                ][:3]
                
                # 후속 질문이 없으면 기본값
                if not response.follow_up_questions:
                    response.follow_up_questions = [
                        "스마트스토어에서 판매할 수 있는 상품에 대해 알려드릴까요?",
                        "온라인 스토어 창업에 관심 있으신가요?",
                        "스마트스토어의 다양한 기능을 알려드릴까요?"
                    ]
            
            except Exception as e:
                logger.error(f"Follow-up question generation error: {e}")
                response.follow_up_questions = [
                    "스마트스토어에서 판매할 수 있는 상품에 대해 알려드릴까요?",
                    "온라인 스토어 창업에 관심 있으신가요?",
                    "스마트스토어의 다양한 기능을 알려드릴까요?"
                ]
            
            yield response
            return

        
        # 세션 히스토리 강제 초기화 방지
        if request.session_id not in chat_histories:
            chat_histories[request.session_id] = ChatHistory(
                messages=[], 
                session_id=request.session_id
            )

        # 이전 대화 컨텍스트 강화
        chat_history = get_chat_history(request.session_id)
        previous_context = ""
        
        if chat_history.messages:
            # 최근 3개 메시지의 전체 컨텍스트 추출
            previous_context = "\n".join([
                f"{msg.role}: {msg.content}" 
                for msg in chat_history.messages[-3:]
            ])

        # 유사도 검색 시 이전 컨텍스트도 함께 검색
        combined_query = f"{previous_context}\n{request.message}"
        similar_faqs = vector_store.query_similar(combined_query)
        #logger.info(f"Similar questions: {[meta['question'] for meta in similar_faqs['metadatas'][0]]}")

        response = ChatResponse(
            answer='',
            sources=[meta['question'] for meta in similar_faqs['metadatas'][0]],
            confidence_score=calculate_confidence(similar_faqs)
        )

        # 이전 대화 내용 가져오기
        chat_history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history.messages[-5:]])

        system_prompt = f"""
        당신은 네이버 스마트스토어 FAQ 챗봇입니다.
        아래 제공된 FAQ 정보와 대화 내용을 토대로 사용자의 질문에 친절하고 정확하게 답변해주세요.

        답변 작성 규칙:
        1. FAQ와 대화 내용을 참고하여 사용자의 상황에 맞는 적절한 답변을 제공하세요.
        2. 답변에 확실한 정보만 포함하고, 불확실한 내용은 제외하세요.
        3. FAQ에 관련 정보가 있다면, 그 내용을 상세히 설명해주세요.
        4. FAQ에 정보가 없다면, 대화 내용을 토대로 최선의 답변을 해주세요.

        이전 대화 내용:
        {chat_history_text}

        참고할 FAQ 정보:
        질문들:
        {[meta['question'] for meta in similar_faqs['metadatas'][0]]}

        답변들:
        {similar_faqs['documents'][0]}

        위 FAQ 정보와 대화 내용을 바탕으로 다음 질문에 답변해주세요:
        {request.message}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]

        completion = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            stream=True
        )

        collected_message = []
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                collected_message.append(content)
                
                # 스트리밍 중 답변 업데이트
                response.answer += content
                yield response

        # 최종 답변 완성
        full_response = ''.join(collected_message)
        response.answer = full_response
        
        # 후속 질문 생성
        response.follow_up_questions = generate_follow_up_questions(
            request.message, 
            full_response
        )

        # 채팅 히스토리 업데이트
        update_chat_history(
            request.session_id, 
            ChatMessage(role="user", content=request.message)
        )
        update_chat_history(
            request.session_id, 
            ChatMessage(role="assistant", content=full_response)
        )

        yield response

    except Exception as e:
        logger.error(f"Chat stream error: {str(e)}")
        error_response = ChatResponse(
            answer=f"오류가 발생했습니다: {str(e)}",
            confidence_score=0.0
        )
        yield error_response
