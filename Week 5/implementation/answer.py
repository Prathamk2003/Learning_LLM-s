from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    convert_to_messages,
)

from langchain_core.documents import Document


# =========================
# Load Environment Variables
# =========================
load_dotenv(override=True)


# =========================
# Configuration
# =========================
MODEL = "llama3.2"

DB_NAME = str(
    Path(__file__).parent.parent / "vector_db"
)

RETRIEVAL_K = 10


# =========================
# Embeddings Model
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Vector Database
# =========================
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)


# =========================
# LLM
# =========================
llm = ChatOllama(
    model=MODEL,
    temperature=0,
)


# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.

You are chatting with a user about Insurellm.

If relevant, use the given context to answer the question.

If you don't know the answer, say so.

Context:
{context}
"""


# =========================
# Fetch Context
# =========================
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant documents for the question.
    """

    return retriever.invoke(question)


# =========================
# Combine User Questions
# =========================
def combined_question(
    question: str,
    history: list[dict] = [],
) -> str:
    """
    Combine previous user questions with current question.
    """

    prior = "\n".join(
        m["content"]
        for m in history
        if m["role"] == "user"
    )

    return prior + "\n" + question


# =========================
# Answer Question with RAG
# =========================
def answer_question(
    question: str,
    history: list[dict] = [],
) -> tuple[str, list[Document]]:
    """
    Generate answer using RAG.
    Returns:
        answer,
        retrieved documents
    """

    # Combine Conversation
    combined = combined_question(
        question,
        history,
    )

    # Retrieve Documents
    docs = fetch_context(combined)

    # Build Context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Create Prompt
    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    # Build Messages
    messages = [
        SystemMessage(content=system_prompt)
    ]

    messages.extend(
        convert_to_messages(history)
    )

    messages.append(
        HumanMessage(content=question)
    )

    # Generate Response
    response = llm.invoke(messages)

    return response.content, docs