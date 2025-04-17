from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
gemini_embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
huggingface_embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

file_path = os.path.join(current_dir, 'docs', 'quit-india-movement.pdf')
loader = PyPDFLoader(file_path)
docs = loader.load()

def create_vector_db(docs, db_name, embedding):
    db_path = os.path.join(db_dir, db_name)
    if not os.path.exists(db_path):
        print(f" ===== Creating new vector store {db_name}... =====\n")
        Chroma.from_documents(docs, embedding, persist_directory=db_path)
    else:
        print(f"Loading existing vector store {db_name}...\n")

def query_vector_db(query, db_name, embedding):
    persistant_dir = os.path.join(db_dir, db_name)
    db = Chroma(persist_directory=persistant_dir, embedding_function=embedding)
    retriever = db.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={'k': 3, 'score_threshold': 0.5}
    )
    print(f" ===== Retrieving documents from {db_name} ===== \n")
    relevant_docs = retriever.invoke(query)
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{GREEN}Document {i}:{END}\n {doc.page_content}\n")
        if doc.metadata:
            print(f"{RED}Metadata: {doc.metadata}{END}\n")

recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
recursive_split_docs = recursive_splitter.split_documents(docs)
create_vector_db(recursive_split_docs, 'chroma_gemini_embedding', gemini_embedding)
create_vector_db(recursive_split_docs, 'chroma_huggingface_embedding', huggingface_embedding)

query = "When quit india movement stared?"
query_vector_db(query, 'chroma_gemini_embedding', gemini_embedding)
query_vector_db(query, 'chroma_huggingface_embedding', huggingface_embedding)


