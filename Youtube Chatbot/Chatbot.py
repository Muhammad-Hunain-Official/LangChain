import os
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def extract_video_id(video_url):
    """YouTube URL ya direct video ID se ID extract karta hai."""

    video_url = video_url.strip()

    # Agar user ne direct 11-character video ID di ho
    if len(video_url) == 11 and all(
        c.isalnum() or c in "_-" for c in video_url
    ):
        return video_url

    parsed_url = urlparse(video_url)
    hostname = parsed_url.hostname or ""
    video_id = None

    if hostname == "youtu.be" or hostname.endswith(".youtu.be"):
        video_id = parsed_url.path.strip("/").split("/")[0]

    elif hostname.endswith("youtube.com"):
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]

        if not video_id:
            parts = parsed_url.path.strip("/").split("/")

            if len(parts) >= 2 and parts[0] in (
                "shorts", "embed", "live"
            ):
                video_id = parts[1]

    if not video_id or len(video_id) != 11:
        raise ValueError("Please enter a valid YouTube URL or video ID.")

    return video_id


def get_transcript(video_id):
    """Video ka English transcript fetch karta hai."""

    youtube_api = YouTubeTranscriptApi()

    transcript_list = youtube_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    if not transcript.strip():
        raise ValueError("The video transcript is empty.")

    return transcript


def format_docs(retrieved_docs):
    """Retrieved chunks ko ek context string mein convert karta hai."""

    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )


def build_rag_chain(video_id):
    """Video ke liye complete RAG chain banata hai."""

    # 1. Transcript
    transcript = get_transcript(video_id)

    # 2. Transcript splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    chunks = splitter.create_documents([transcript])

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. FAISS vector store
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # 5. Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 6. Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2
    )

    # 7. Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful YouTube video assistant.

Answer the user's question using the video transcript context.

Instructions:
- Give a clear and easy-to-understand answer.
- Use the supplied context to answer.
- Do not invent information.
- If the context does not contain enough information,
  say: "The answer was not found in the video transcript."

Context:
{context}

Question:
{question}

Answer:"""
    )

    # 8. Context + question
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    # 9. Final chain
    parser = StrOutputParser()

    main_chain = (
        parallel_chain
        | prompt
        | llm
        | parser
    )

    return main_chain


def ask_question(chain, question):
    """RAG chain se answer leta hai."""

    return chain.invoke(question)