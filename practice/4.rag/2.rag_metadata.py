from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

RED = '\033[91m'
GREEN = '\033[92m'
END = '\033[0m'

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, 'db', 'chromadb_metadata')
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

def initiate_vector_store():
    files_dir = os.path.join(current_dir, 'docs')

    if not os.path.exists(persistent_dir):
        files = [f for f in os.listdir(files_dir) if f.endswith('.pdf')]

        documents = []

        for file in files:
            file_path = os.path.join(files_dir, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata = {'source': file}
                documents.append(doc)
        
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        doc_chunks = splitter.split_documents(documents)
        Chroma.from_documents(doc_chunks, embeddings, persist_directory=persistent_dir)
    else:
        print("Loading existing vector store...\n")

initiate_vector_store()

query = "What was Gandhi 'mantra' in his quit India movement speech?"

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': 3, 'score_threshold': 0.5}
)

relevant_docs = retriever.invoke(query)
for i, doc in enumerate(relevant_docs, 1):
    print(f"{GREEN}Document {i}:{END}\n {doc.page_content}\n")
    if doc.metadata:
        print(f"{RED}Metadata: {doc.metadata}{END}\n")
