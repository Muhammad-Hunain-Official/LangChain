from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate   
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI( model="gemini-2.5-flash",temperature=0.2, max_output_tokens=512)
prompt = PromptTemplate(
    template = "Write the summary of the following text:\n{text}\n\nSummary:",
    input_variables = ["text"],
    )
parser = StrOutputParser()
loader = ("Questions.txt")
docs = loader.load()
chain = prompt | model | parser 
print(chain.invoke({'text':docs[0].page_content}))
