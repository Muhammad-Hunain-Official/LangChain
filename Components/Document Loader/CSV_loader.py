from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader = CSVLoader(file_path="students.csv")
docs = loader.load()
# print(docs[0].page_content)
text = "\n".join(doc.page_content for doc in docs)
prompt = PromptTemplate(
    template="Answer The Following Question: {Question} based on the following content: {text}",
    input_variables=["Question", "text"]
)
model = ChatGoogleGenerativeAI(
      model="gemini-3.8-flash"
)
parser = StrOutputParser()
chain   = prompt | model | parser
print(chain.invoke({'Question': "What Was the gpa of a student name saad ", 'text': text}))
