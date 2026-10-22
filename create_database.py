
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load PDF document

# Use the relative path matching your workspace directory
data = PyPDFLoader("document loader/deeplearning.pdf")


docs = data.load()
print(f"PDF loaded successfully! Pages: {len(docs)}")

# 2. Split document into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# 3. Load FREE local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Free embedding model loaded successfully!")

# 4. Store chunks and embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Database created successfully!")
print("Your PDF is stored in the local Chroma database.")