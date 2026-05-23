import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
import os

st.set_page_config(page_title="Epitech Chroma RAG")
st.title("Chatbot")

# 초기화
llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")

# DB 경로 지정
CHROMA_DB_DIR = os.path.join(os.getcwd(), "epitech_chroma_db")

# 벡터 DB 생성, 로드
@st.cache_resource
def init_chroma_db():
    # English Reference Documents
    raw_documents = [
        "EPITECH Paris Campus is a leading IT and computer science higher education institution in France.",
        "The 2026 Summer Global Program begins with a departure from Incheon to Paris on June 27th.",
        "The core AI Hackathon project runs for 3 days, from July 8th to July 10th.",
        "The final presentation and evaluation of the Hackathon will take place on July 10th at 2:00 PM, followed by an award ceremony.",
        "The program includes basic French language classes and Paris cultural experience activities to help students adapt."
    ]
    
    # 백터 DB 객체 생성
    vector_store = Chroma.from_texts(
        texts=raw_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    return vector_store

# 백터 DB 객체 확보
vector_db = init_chroma_db()

# 대화 기록 관리 및 랜더링
if "messages" not in st.session_state:
    st.session_state.messages = []

# 누적된 대화 기록 순회 
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# 사용자 입력 및 문서 검색
if user_input := st.chat_input("Ask about the EPITECH program..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 사용자의 질문을 벡터로 바꿔 DB 내에서 사장 유사도가 높은 문장 2개를 검색해서 가져옴
    retrieved_docs = vector_db.similarity_search(user_input, k=2)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 가이드라인
    rag_prompt = f"""[Context]
{context}

[Question]
{user_input}

Instructions: Answer the question based strictly on the provided [Context]. Keep your answer concise and factual. Do NOT include the <think> tag or reasoning process; provide only the final answer in English."""

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from ChromaDB and generating answer..."):
            response = llm.invoke([HumanMessage(content=rag_prompt)])
            
            final_answer = response.content
            if "</think>" in final_answer:
                final_answer = final_answer.split("</think>")[-1].strip()
                
            st.markdown(final_answer)
            
    st.session_state.messages.append(AIMessage(content=final_answer))