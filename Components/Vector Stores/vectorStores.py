print("HELLO")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

print("Imports done")

load_dotenv()
print("ENV loaded")

doc1 = Document(     page_content="Football is a team sport played between two teams of eleven players.",     metadata={"team": "RCB"} )
doc2 = Document(     page_content="Basketball is a sport played between two teams of five players each.",     metadata={"team": "LAL"} )
doc3 = Document(     page_content="Tennis is a sport played between two players or two pairs of players.",     metadata={"team": "NYG"} )
doc4 = Document(     page_content="Tennis is a sport played between two players or two pairs of players.",     metadata={"team": "NYG"} ) 
doc5 = Document(     page_content="Foosball is a sport played between two teams of five players each.",     metadata={"team": "Pal"} )
docs = [doc1, doc2, doc3, doc4, doc5]
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

print("Embeddings created")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_test_db",
    collection_name="sample",
)
print("API key loaded:", bool(os.getenv("GEMINI_API_KEY")))

print("Vector store created")

vector_store.add_documents(docs)

print("Documents added")

results = vector_store.similarity_search(
    "Which sport is played between two players or two pairs of players?",
    k=1
)

print("Search completed")
print(results)