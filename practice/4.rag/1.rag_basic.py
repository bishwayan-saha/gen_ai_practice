from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

RED = '\033[91m'
END = '\033[0m'
GREEN = '\033[92m'

current_path = os.path.dirname(os.path.abspath(__file__))
persistant_path = os.path.join(current_path, 'db', 'chromadb')
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')


def initiate_vector_store(file_name: str = 'quit-india-movement.pdf'):

    file_path = os.path.join(current_path, 'docs', file_name)

    if not os.path.exists(persistant_path):
        print("Creating new vector store...\n")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_chunks = splitter.split_documents(docs)

        print(f'Number of document chunks {len(doc_chunks)}')


        db = Chroma.from_documents(doc_chunks, embeddings, persist_directory= persistant_path)
        print(f'Vector store created at {persistant_path}\n')
    else:
        print("Loading existing vector store...\n")

initiate_vector_store()

query = input("Enter query: ")
db = Chroma(persist_directory=persistant_path, embedding_function=embeddings)
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 3, 'score_threshold': 0.5}
)
print(" ===== Retrieving documents ===== \n")

relevant_docs = retriever.invoke(query)

print(" ===== Relevant documents ===== \n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"{GREEN}Document {i}:{END}\n {doc.page_content}\n")
    if doc.metadata:
        print(f"{RED} Metadata: {doc.metadata} {END}\n")

model.

