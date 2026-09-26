from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PyPDFLoader("doc.pdf") 
docs = loader.load()

text = """
Technology has become an important part of our daily lives. People use computers, smartphones, and the internet for communication, education, business, entertainment, and many other activities. Students can attend online classes, search for information, and learn new skills from anywhere. Similarly, businesses use technology to manage their operations and communicate with customers more effectively.

Artificial Intelligence is one of the fastest-growing areas of technology. AI allows computers to perform tasks that normally require human intelligence, such as understanding language, recognizing images, analyzing data, and making predictions. Machine learning is a major part of AI, where computers learn patterns from data and improve their performance without being explicitly programmed for every task.

Learning programming and data-related skills can provide many opportunities for students. Python is widely used for data analysis, automation, artificial intelligence, and machine learning. SQL is useful for working with databases, while tools such as Pandas, NumPy, and Power BI help users analyze and visualize information. With regular practice and practical projects, students can gradually build strong technical skills and prepare themselves for future careers in technology.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
# result = splitter.split_documents(docs)
# print(result)

chunk = splitter.split_text(text)
print(len(chunk))
print(chunk)