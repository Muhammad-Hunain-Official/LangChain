from langchain_text_splitters import CharacterTextSplitter

text = """

Artificial Intelligence

Artificial Intelligence, commonly known as AI, is a branch of computer science that focuses on creating systems that can perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images, solving problems, making decisions, and learning from experience. AI is now used in many areas of modern technology.

Machine Learning

Machine Learning is an important part of Artificial Intelligence. Instead of programming every rule manually, machine learning algorithms learn patterns from data. The more useful and relevant data a model receives, the better it can learn certain patterns. Machine learning is commonly divided into supervised learning, unsupervised learning, and reinforcement learning.

Supervised Learning

In supervised learning, a model learns from labeled data. Each training example contains an input and a correct output. The model studies the relationship between inputs and outputs and then uses that knowledge to make predictions on new data. Classification and regression are two common supervised learning tasks.

Unsupervised Learning

Unsupervised learning works with data that does not have predefined labels. The algorithm tries to discover hidden patterns or structures in the data. Clustering is a common example of unsupervised learning. It can be used to group customers based on their behavior, identify similar documents, or discover patterns in large datasets.

Deep Learning

Deep Learning is a specialized area of machine learning that uses neural networks with multiple layers. These networks can automatically learn complex features from large amounts of data. Deep learning has achieved impressive results in computer vision, speech recognition, natural language processing, and recommendation systems.

Natural Language Processing

Natural Language Processing, or NLP, focuses on enabling computers to understand and work with human language. NLP techniques are used in chatbots, translation systems, sentiment analysis, text summarization, search engines, and virtual assistants. Modern language models can process large amounts of text and generate human-like responses.

Computer Vision

Computer Vision allows computers to understand information from images and videos. A computer vision system can detect objects, recognize faces, classify images, and analyze visual scenes. It is used in medical imaging, autonomous vehicles, security systems, manufacturing, and many other applications.

Generative AI

Generative AI is a type of artificial intelligence that can create new content. Depending on the model, this content may include text, images, audio, video, or computer code. Large language models are examples of generative AI systems that can generate text based on instructions provided by users.

Large Language Models

Large Language Models, or LLMs, are AI models trained on large collections of text. They learn statistical patterns in language and can perform tasks such as answering questions, summarizing documents, translating text, generating code, and assisting with research. LLMs are now widely used in modern AI applications.

Retrieval Augmented Generation

Retrieval Augmented Generation, commonly called RAG, combines information retrieval with language generation. Instead of relying only on information learned during model training, a RAG system retrieves relevant information from external documents and provides that information to the language model. This approach can help an AI system answer questions using a specific knowledge base.

Document Loaders

In LangChain, document loaders are used to load data from different sources. Examples include text files, PDF files, CSV files, web pages, and many other formats. A document loader converts the source data into a format that can be processed by other LangChain components.

Text Splitters

Text splitters are used to divide large documents into smaller pieces called chunks. Splitting documents is important because language models have limits on how much text they can process at one time. Smaller chunks can also make information retrieval more efficient.

Chunk Size and Chunk Overlap

Chunk size determines approximately how much text should be placed inside each chunk. A larger chunk contains more information, while a smaller chunk contains less information. Chunk overlap means that some text from the previous chunk is repeated in the next chunk. Overlap helps preserve context when important information appears near the boundary between two chunks.

Embeddings

Embeddings represent text as numerical vectors. Text with similar meanings tends to have similar vector representations. Embeddings are commonly used in semantic search and retrieval systems. In a RAG application, document chunks can be converted into embeddings and stored in a vector database.

Vector Databases

A vector database stores numerical representations of data and allows applications to search for similar vectors. When a user asks a question, the system can convert the question into an embedding and search the vector database for relevant document chunks.

AI Applications

Artificial Intelligence is being used in healthcare, finance, education, transportation, customer service, cybersecurity, and software development. AI systems can help automate repetitive tasks, analyze large datasets, identify patterns, and assist people in making decisions.

Future of AI

Artificial Intelligence continues to develop rapidly. Future AI systems may become more capable of understanding complex information, working with multiple types of data, and assisting humans with a wider range of tasks. Learning the fundamentals of Python, machine learning, data processing, and AI frameworks can provide a strong foundation for building AI applications."""


splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
result = splitter.split_text(text)

print(result)