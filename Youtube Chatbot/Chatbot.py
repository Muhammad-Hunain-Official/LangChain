from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# 
load_dotenv()
video_id = "Gfr50f6ZBvo"

# Step 1: Fetch YouTube transcript
try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    print("Transcript fetched successfully!")
    print(transcript)

except TranscriptsDisabled:
    print("Transcript is disabled for this video.")
    raise SystemExit

except Exception as e:
    print(f"Transcript fetching failed: {e}")
    raise SystemExit

#
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,chunk_overlap=100)
chunks = splitter.create_documents([transcript])   
print(len(chunks))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store= FAISS.from_documents(chunks,embeddings) 
# print(vector_store)
