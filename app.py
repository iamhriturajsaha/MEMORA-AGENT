import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# PAGE
st.set_page_config(page_title="MEMORA AI Agent", layout="wide")
st.title("🧠 MEMORA — Intelligent Document Assistant")

# KEY CHECK
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("❌ GROQ_API_KEY not set.")
    st.stop()

# SIDEBAR
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# LOAD EMBEDDINGS
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
embeddings = load_embeddings()

# BUILD VECTORSTORE
def build_vectorstore(files):
    docs = []
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.split_documents(docs)
    docs = [d for d in docs if d.page_content.strip()]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# SESSION STATE
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# PROCESS BUTTON
if uploaded_files:
    if st.sidebar.button("🚀 Process Documents"):
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = build_vectorstore(uploaded_files)
        st.sidebar.success("✅ Documents processed!")

# LLM
@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
llm = load_llm()

# PROMPT
prompt = ChatPromptTemplate.from_template(
    """You are MEMORA, an AI assistant.
Use ONLY the provided context to answer.

Context:
{context}

Question:
{question}

Answer clearly and concisely."""
)

# HELPERS
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def get_question(x):
    return x["question"]

# MEMORY
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# CHAT UI
st.subheader("💬 Chat with your documents")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
user_query = st.chat_input("Ask something about your documents...")
if user_query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process documents first.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnableLambda(get_question),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    qa_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": "streamlit_user"}},
            )
            st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
