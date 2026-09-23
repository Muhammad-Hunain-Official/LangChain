from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("doc.pdf")
docs = loader.load()

print(len(docs))