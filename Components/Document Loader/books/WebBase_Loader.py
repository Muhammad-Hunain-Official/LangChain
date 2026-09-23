from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://zerolifestyle.co/pages/earbuds-zbuds?srsltid=AU7gw4VExsTxbEj44xPEeb-t1lotGWe8yi0CEnNYbTQKNYAL1Ws4WwuY")
docs = loader.load()
print(len(docs))