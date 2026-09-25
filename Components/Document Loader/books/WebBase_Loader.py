from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = WebBaseLoader("https://zerolifestyle.co/pages/earbuds-zbuds?srsltid=AU7gw4VExsTxbEj44xPEeb-t1lotGWe8yi0CEnNYbTQKNYAL1Ws4WwuY")
docs = loader.load()

prompt = PromptTemplate(
    template="Answer The Following Question: {Question} based on the following content: {text}",
    input_variables=["Question", "text"]
)
model = ChatGoogleGenerativeAI(
      model="gemini-3.8-flash"
)
parser = StrOutputParser()


chain = prompt | model | parser 
print(chain.invoke({'Question' : "What is the main topic of the webpage?",'text':docs[0].page_content}))