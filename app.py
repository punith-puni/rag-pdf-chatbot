import streamlit as st
import tempfile
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os
# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CYBERPUNK CSS
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0B0F19;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #07111f;
    border-right: 1px solid #00F5FF;
}

/* Main glowing container */
.main-wrapper {
    border: 2px solid #00F5FF;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0px 0px 20px #00F5FF;
    margin-top: 10px;
}

/* Title */
.main-title {
    text-align: center;
    color: #00F5FF;
    font-size: 42px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #9ecfff;
    margin-bottom: 20px;
}

/* Chat messages */
.stChatMessage {
    border-radius: 15px;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 2px solid #00F5FF;
    border-radius: 15px;
    padding: 10px;
}

/* Success box */
.stSuccess {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# GROQ CLIENT
# ==========================================
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🤖 AI Knowledge Assistant")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.markdown("---")

    st.markdown("""
    ### Tech Stack

    - Groq
    - Llama 3
    - FAISS
    - LangChain
    - Streamlit
    - Sentence Transformers
    """)

# ==========================================
# MAIN UI
# ==========================================

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🤖 AI Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Semantic Search • RAG • Llama 3 • FAISS</div>',
    unsafe_allow_html=True
)

# ==========================================
# PDF PROCESSING
# ==========================================

if uploaded_file is not None and st.session_state.vectorstore is None:

    with st.spinner("🤖 Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            texts,
            embeddings
        )

        st.session_state.vectorstore = vectorstore

    st.success("✅ PDF Ready!")

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask a question about your PDF..."
)

# ==========================================
# RAG + GROQ
# ==========================================

if question and st.session_state.vectorstore:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    docs = st.session_state.vectorstore.similarity_search(
        question,
        k=2
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

    except Exception as e:

        answer = f"Error: {str(e)}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

st.markdown("</div>", unsafe_allow_html=True)
