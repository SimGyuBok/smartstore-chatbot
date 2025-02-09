import chromadb
import pandas as pd
import pickle
from app.config import get_settings
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY
        )
        
        self.collection = self._get_or_create_collection()
        self.vectorizer = TfidfVectorizer(max_features=5000)

        # FAQ 데이터가 이미 로드되어 있는지 확인
        if self.collection.count() == 0:
            # 데이터가 없으면 로드 및 인덱싱 수행
            self.load_and_index_data()
        else:
            # 데이터가 이미 있으면 vectorizer만 로드
            self.load_vectorizer()

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection("smartstore_faq")
        except:
            return self.client.create_collection("smartstore_faq")

    def load_and_index_data(self):
        logger.info("Starting to load FAQ data...")
        try:
            with open("data/final_result.pkl", "rb") as f:
                faq_data = pickle.load(f)
            
            questions = []
            answers = []
            
            for question, answer in faq_data.items():
                answer = answer.split('위 도움말이 도움이 되었나요?')[0].strip()
                questions.append(question)
                answers.append(answer)
            
            logger.info(f"Prepared {len(questions)} FAQ pairs")
            
            # TF-IDF 벡터화
            self.vectorizer.fit(questions)
            question_vectors = self.vectorizer.transform(questions)

            for i in range(len(questions)):
                self.collection.add(
                    documents=[answers[i]],
                    metadatas=[{
                        "question": questions[i],
                        "question_vector": str(question_vectors[i].toarray()[0].tolist())
                    }],
                    ids=[str(i)]
                )

            logger.info("Completed indexing all FAQ data")
            
        except Exception as e:
            logger.error(f"Error in load_and_index_data: {str(e)}")
            raise

    def query_similar(self, query: str, n_results: int = 3):
        try:
            # 쿼리 전처리: 불필요한 특수문자 제거, 길이 제한
            clean_query = re.sub(r'[^\w\s가-힣]', '', query)
            clean_query = clean_query[:200]  # 쿼리 길이 제한

            # 벡터 검색 로직
            query_vector = self.vectorizer.transform([clean_query]).toarray()
            
            # 검색 결과 확장
            initial_results = self.collection.query(
                query_texts=[clean_query], 
                n_results=n_results * 5,  
                include=['documents', 'metadatas', 'distances']
            )

            # 추가 유사도 필터링 로직
            similarities = []
            for meta in initial_results['metadatas'][0]:
                if 'question_vector' in meta:
                    question_vec = eval(meta['question_vector'])
                    similarity = cosine_similarity(query_vector, [question_vec])[0][0]
                    similarities.append(similarity)
                else:
                    similarities.append(0)

            # 유사도 임계값 조정
            filtered_indices = [
                i for i, sim in enumerate(similarities) 
                if sim > 0.2  # 유사도 임계값 조정
            ]

            # 결과 필터링
            filtered_results = {
                'documents': [[initial_results['documents'][0][i] for i in filtered_indices]],
                'metadatas': [[initial_results['metadatas'][0][i] for i in filtered_indices]],
                'distances': [[initial_results['distances'][0][i] for i in filtered_indices]]
            }

            return filtered_results if filtered_results['documents'][0] else self._get_default_faq_response()

        except Exception as e:
            logger.error(f"검색 중 오류 발생: {e}")
            return self._get_default_faq_response()

    def _get_default_faq_response(self):
        return {
            "documents": [["스마트스토어와 관련된 일반적인 질문에 대해 도와드리겠습니다. 구체적인 질문을 해주세요."]],
            "metadatas": [[{"question": "기본 FAQ 응답"}]],
            "distances": [[ 1.0 ]]
        }

    def load_vectorizer(self):
        try:
            # 저장된 질문 데이터 로드
            #print(self.collection.get())
            #data = self.collection.get()
            #print(type(data))  # 데이터 타입 확인
            #print(data)  
            questions = [meta['question'] for meta in self.collection.get().get('metadatas', []) if isinstance(meta, dict)]
            
            # 질문 데이터로 vectorizer 학습
            self.vectorizer.fit(questions)
            
        except Exception as e:
            logger.error(f"Error in load_vectorizer: {str(e)}")
            raise


