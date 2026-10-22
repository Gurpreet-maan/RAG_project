
import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load the free local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the existing ChromaDB database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 3. Create the retriever using MMR strategy
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# 4. Initialize the Mistral chat model
llm = ChatMistralAI(
    model="codestral-2508",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

# 5. Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
        Use only the provided context to answer the question.
        If the context is not sufficient to answer the question,
        say: "I could not find the answer in the provided context."
        """
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

print("RAG system created!")
print("Press 0 to exit")

# 6. Ask questions continuously
while True:
    query = input("You: ")

    if query.strip() == "0":
        break

    # Retrieve relevant document chunks
    docs = retriever.invoke(query)

    # Combine retrieved chunks into context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Fill the prompt variables
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate the answer using Mistral
    response = llm.invoke(final_prompt)

    print(f"Answer: {response.content}")
