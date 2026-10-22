import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="BookMind AI",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------

st.markdown("""
<style>
.stApp {
    background-color: #fcfcfd;
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.hero h1 {
    color: #0f172a !important;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 8px;
}
.hero p {
    color: #475569 !important;
    font-size: 16px;
}
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #0f172a !important;
    line-height: 1.6;
}
[data-testid="stFileUploader"] {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px;
}
.stButton > button {
    border-radius: 6px;
    min-height: 42px;
    font-weight: 500;
}
[data-testid="stChatInput"] {
    border-radius: 6px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
@st.cache_resource
def load_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY missing. Add it to your .env file."
        )

    return ChatMistralAI(
        model="codestral-2508",
        mistral_api_key=api_key
    )

# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are BookMind AI, a helpful book assistant.

Answer the user's question using only the provided context.
Do not make up facts that are not in the context.
If the context is insufficient, say:
"I could not find the answer in the uploaded book."

Be clear, helpful, and easy to understand."""
    ),
    (
        "human",
        "Book context:\n{context}\n\nQuestion: {question}"
    )
])

# ---------------- SESSION STATE ----------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>📚 BookMind AI</h1>
    <p>Your personal AI reading assistant.</p>
    <p>Upload a PDF book, ask questions, and explore its content
    using Retrieval-Augmented Generation (RAG).</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📖 Your Library")
    st.write("Upload a PDF to begin.")

    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"],
        help="Select a PDF file from your computer."
    )

    process_button = st.button(
        "📚 Process Book",
        use_container_width=True,
        type="primary"
    )

    if st.session_state.book_name:
        st.divider()
        st.subheader("Current Book")
        st.write(f"**{st.session_state.book_name}**")
        st.caption(
            f"{st.session_state.page_count} pages · "
            f"{st.session_state.chunk_count} chunks"
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- PROCESS UPLOADED BOOK ----------------

if process_button:
    if uploaded_file is None:
        st.sidebar.warning("Please upload a PDF first.")

    else:
        try:
            with st.spinner("Reading and processing your book..."):

                # Save uploaded PDF temporarily
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = os.path.join(
                        temp_dir, uploaded_file.name
                    )

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Load PDF
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()

                    if not documents:
                        raise ValueError(
                            "No readable pages found in this PDF."
                        )

                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(documents)

                    if not chunks:
                        raise ValueError(
                            "No text could be extracted from this PDF."
                        )

                    # Load local embeddings
                    embeddings = load_embeddings()

                    # Create an isolated temporary vector database
                    db_dir = tempfile.mkdtemp(
                        prefix="bookmind_chroma_"
                    )

                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=db_dir
                    )

                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )

                    # Save active book in session state
                    st.session_state.retriever = retriever
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.page_count = len(documents)
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.messages = []

                    # Keep the temporary database alive
                    st.session_state.db_dir = db_dir

            st.sidebar.success("Book processed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Could not process the PDF: {e}")

# ---------------- MAIN CONTENT ----------------

if st.session_state.retriever is None:

    st.info("👈 Upload a PDF book from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 Upload")
        st.write("Choose a PDF book from your computer.")

    with col2:
        st.subheader("🔎 Retrieve")
        st.write("Find relevant passages with MMR retrieval.")

    with col3:
        st.subheader("💬 Ask")
        st.write("Ask questions grounded in your book.")

else:

    st.subheader("💬 Chat with your book")
    st.caption(
        f"Currently reading: {st.session_state.book_name}"
    )
   # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept a new question
    question = st.chat_input(
        "Ask anything about your uploaded book..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your book..."):

                    # Retrieve relevant chunks
                    docs = st.session_state.retriever.invoke(
                        question
                    )

                    if not docs:
                        answer = (
                            "I could not find the answer "
                            "in the uploaded book."
                        )
                        sources = []

                    else:
                        # Include page information in context
                        context_parts = []

                        for doc in docs:
                            page = doc.metadata.get("page")
                            page_label = (
                                f"Page {page + 1}"
                                if isinstance(page, int)
                                else "Page unknown"
                            )

                            context_parts.append(
                                f"[{page_label}]\n{doc.page_content}"
                            )

                        context = "\n\n".join(context_parts)

                        # Generate answer
                        llm = load_llm()

                        final_prompt = prompt.invoke({
                            "context": context,
                            "question": question
                        })

                        response = llm.invoke(final_prompt)
                        answer = response.content

                        # Unique page references
                        sources = sorted({
                            doc.metadata["page"] + 1
                            for doc in docs
                            if isinstance(
                                doc.metadata.get("page"), int
                            )
                        })

                st.markdown(answer)

                if sources:
                    with st.expander("📑 Retrieved pages"):
                        st.write(
                            ", ".join(
                                f"Page {page}" for page in sources
                            )
                        )

                # Save answer for chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                error_message = f"Something went wrong: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })