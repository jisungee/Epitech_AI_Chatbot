import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import os

# ==========================================
# 1. 페이지 및 레이아웃 설정
# ==========================================
# 웹 브라우저 탭 제목 및 화면 레이아웃을 좌우로 넓게 설정
st.set_page_config(page_title="Epitech Folder RAG", layout="wide")
st.title("Chatbot")


# ==========================================
# 2. AI 모델 및 임베딩 엔진 초기화
# ==========================================
# ChatOllama: 로컬 Ollama 환경에서 DeepSeek-R1 추론 모델을 호출
# temperature=0.1: AI의 무작위성을 낮춰 주어진 문서에만 기반해 사실적인 답변을 하도록 유도
llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)

# OllamaEmbeddings: 텍스트 단어/문장을 컴퓨터가 연산할 수 있는 백터로 변환
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")


# ==========================================
# 3. 물리 경로 정의 (문서 폴더 및 벡터 DB 저장 폴더)
# ==========================================
# 원본 PDF 파일들을 미리 보관해 둘 로컬 'data' 폴더의 경로 설정
DATA_DIR = os.path.join(os.getcwd(), "data")

# 문장 조각들과 벡터 데이터가 실제로 영구 저장될 ChromaDB 디렉토리 경로를 설정
CHROMA_DB_DIR = os.path.join(os.getcwd(), "epitech_chroma_db")

# 사용자가 직접 생성하지 않았을 경우를 대비해 프로젝트 루트에 'data' 폴더를 자동 생성
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# ==========================================
# 4. data 폴더 내 PDF 자동 감지 및 ChromaDB 인덱싱 함수
# ==========================================
# @st.cache_resource: 스트림릿 화면이 리프레시될 때마다 DB를 처음부터 다시 파싱·빌드하는 현상 방지
# 데이터베이스 메모리 로드 객체를 캐싱하여 최초 1회만 무거운 연산을 실행하도록 통제
@st.cache_resource
def init_folder_vector_db():
    # 외부 파일이 없을 때를 대비한 백업용 기본 고정 지식 데이터(프로그램 일정 가이드라인)
    raw_texts = [
        "EPITECH Paris Campus is a leading IT and computer science higher education institution in France.",
        "The 2026 Summer Global Program begins with a departure from Incheon to Paris on June 27th.",
        "The core AI Hackathon project runs for 3 days, from July 8th to July 10th.",
        "The final presentation and evaluation of the Hackathon will take place on July 10th at 2:00 PM.",
        "The program includes basic French language classes and Paris cultural experience activities."
    ]
    
    # 4-1. 먼저 고정형 기본 텍스트 지식들을 벡터화하여 초기 Chroma 벡터 저장소 생성
    vector_store = Chroma.from_texts(
        texts=raw_texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # 4-2. 지정된 data 폴더 내부에서 확장자가 '.pdf'로 끝나는 실제 문서들을 스캔
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    
    # 만약 폴더 내부에 PDF 파일이 한 개 이상 존재한다면 자동 추출 파이프라인 가동
    if pdf_files:
        all_pdf_documents = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(DATA_DIR, pdf_file)
            try:
                # [PDF to Text 단계]: PyPDFLoader가 문서 페이지들을 순차적으로 열어 텍스트 데이터화
                loader = PyPDFLoader(pdf_path)
                all_pdf_documents.extend(loader.load())
            except Exception as e:
                # 특정 PDF 파일 파싱 실패 시 프로그램이 깨지지 않도록 예외 로그만 터미널에 띄움
                print(f"[{pdf_file}] 로드 실패: {e}")
        
        # 텍스트 추출에 성공한 문서 개체가 존재한다면 분할 처리 시작
        if all_pdf_documents:
            # [Text Chunking 단계]: 대형 문서를 통째로 넣으면 AI가 문맥을 잃거나 하드웨어 버퍼가 터짐
            # chunk_size=500: 가독성을 저해하지 않는 단락 단위 기준 500자 내외로 조각냄
            # chunk_overlap=50: 잘린 문장 간 문맥 단절을 예방하기 위해 앞뒤 조각이 50자씩 겹치도록 설계
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(all_pdf_documents)
            
            # [ChromaDB 누적 추가]: 분할 공정을 마친 조각들을 벡터로 임베딩하여 로컬 크로마 디렉토리에 추가 저장
            vector_store.add_documents(chunks)
            
    return vector_store, len(pdf_files)

# 가동 시 자동으로 data 폴더를 전수 검사하여 벡터 DB 인프라 구축
vector_db, detected_pdf_count = init_folder_vector_db()


# ==========================================
# 사이드바 대시보드 (학습 상태 및 대화 제어)
# ==========================================
st.sidebar.header("시스템 모니터")
# 현재 서버가 인식하고 데이터베이스에 동기화 완료한 물리 PDF의 수량을 상시 노출
st.sidebar.info(f"현재 탐지된 data 폴더 내 PDF: {detected_pdf_count}개")
st.sidebar.caption(f"DB 저장 경로: {CHROMA_DB_DIR}")

# 새 문서 동기화 버튼: 사용자가 프로그램 구동 중에 data 폴더에 새로운 PDF 파일을 넣었을 경우 대응
if st.sidebar.button("새 문서 다시 동기화", use_container_width=True):
    # 캐시 메모리를 강제로 클리어하여 init_folder_vector_db() 함수가 폴더를 처음부터 새로 스캔하도록 함
    st.cache_resource.clear()
    st.rerun()

# 대화 기록 초기화 버튼: 대화 이력 배열을 완전 공백으로 밀어버리고 첫 화면으로 복귀
if st.sidebar.button("대화 기록 초기화", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# ==========================================
# 5. 대화 기록 세션 상태 초기화 및 렌더링
# ==========================================
# 최초 진입 시 대화 기록을 보관할 세션 공간(`messages` 배열)이 없으면 빈 저장소를 선언
if "messages" not in st.session_state:
    st.session_state.messages = []

# 화면이 리프레시될 때마다 과거 대화 히스토리 말풍선들을 순서대로 화면에 다시 그려 복원
for message in st.session_state.messages:
    # 랭체인 메시지 객체 구조(HumanMessage, AIMessage)를 파악하여 'user' 혹은 'assistant' 레이아웃을 바인딩
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)


# ==========================================
# 6. 사용자 입력 및 ChromaDB RAG 연동 질의응답
# ==========================================
# 화면 하단에 챗 전용 입력 바를 렌더링하고 유저의 텍스트가 바인딩되면 즉시 아래 스크립트를 가동
if user_input := st.chat_input("Ask anything about documents in 'data' folder..."):
    # 사용자가 던진 질문을 화면 말풍선에 출력하고 대화 기록(Session State) 리스트에 영구 누적
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 6-1. [RAG 핵심 검색 - Retrieval]: 사용자의 질문을 벡터로 즉석 변환하여
    # 크로마 하드디스크 DB 내 문장 조각 중 질문의 의미적 의도와 가장 밀접한 탑 3개(k=3) 추출
    retrieved_data = vector_db.similarity_search(user_input, k=3)
    
    # 찾아온 조각들의 내부 원본 텍스트 내용(page_content)만 분리하여 한 줄씩 결합한 대형 컨텍스트 지식을 형성
    context = "\n".join([doc.page_content for doc in retrieved_data])
    
    # 6-2. [프롬프트 엔지니어링 - Prompt Configuration]:
    # 무작위 추론을 차단하고 오직 크로마DB에서 검출해 온 문서 정보({context}) 속에서만 팩트 기반 답변을 하도록 규칙 설정
    rag_prompt = f"""[Context]
{context}

[Question]
{user_input}

Instructions: Answer the question based strictly on the provided [Context]. Keep your answer concise, precise, and factual. Do NOT include the <think> tag or any reasoning process; provide only the final answer in English."""

    # 6-3. [AI 답변 생성 단계]
    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB and generating answer..."):
            # 정제된 템플릿 프롬프트를 랭체인 표준 규격 메시지에 태워 DeepSeek-R1로 전송
            response = llm.invoke([HumanMessage(content=rag_prompt)])
            
            final_answer = response.content
            # DeepSeek-R1 특유의 추론 루프 태그(<think>...</think>)가 텍스트 외부에 그대로 묻어 나왔을 경우
            # split을 사용하여 오직 사용자가 읽어야 할 순수 최종 결론 답변 본문만 자름
            if "</think>" in final_answer:
                final_answer = final_answer.split("</think>")[-1].strip()
                
            # 정제 가공이 완료된 지식 답변을 UI 말풍선 화면에 출력
            st.markdown(final_answer)
            
            # 6-4. [출처 시각화 디버거 - Traceability]:
            # AI가 답변하는 과정에서 투명성을 보장하기 위해 크로마 DB에서 끄집어낸 파일 원본명과 PDF 실제 페이지 번호를 하단에 표기
            with st.expander("문서 참조 출처 확인 (ChromaDB Retrieval Result)"):
                for idx, doc in enumerate(retrieved_data, start=1):
                    # 경로에서 파일 명칭만 축약 추출
                    source_name = os.path.basename(doc.metadata.get('source', '기본 지식'))
                    # PDF 로더 내부 인덱스(0번부터 시작) 보정을 위해 휴먼 뷰 기준 +1 페이지 처리
                    page_num = doc.metadata.get('page', 0) + 1 if 'page' in doc.metadata else '-'
                    st.markdown(f"**[{idx}] {source_name}** (Page: {page_num})")
                    st.caption(doc.page_content)
            
    # AI 정답 텍스트를 세션 대화 저장소에 최종 인계하여 대화 종료
    st.session_state.messages.append(AIMessage(content=final_answer))