from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
import os

GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

file_path = os.path.join(current_dir, 'docs', 'quit-india-movement.pdf')
loader = PyPDFLoader(file_path)
docs = loader.load()

def create_vector_db(docs, db_name):
    db_path = os.path.join(db_dir, db_name)
    if not os.path.exists(db_path):
        print(f" ===== Creating new vector store {db_name}... =====\n")
        Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    else:
        print(f"Loading existing vector store {db_name}...\n")

def query_vector_db(query, db_name):
    persistant_dir = os.path.join(db_dir, db_name)
    db = Chroma(persist_directory=persistant_dir, embedding_function=embeddings)
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

# 1. Character based splitting
# splits text into chunks based on a predefined number of characters
# Useful for consistent chunk size regardless of content structure
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
char_split_docs = char_splitter.split_documents(docs)
create_vector_db(char_split_docs, 'chroma_char_split')


# 2. Sentence based splitting
# splits text into chunks based on sentence boundaries
# Useful for maintaining semantic coherence within chunks
sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000, chunk_overlap=10)
sentence_split_docs = sentence_splitter.split_documents(docs)
create_vector_db(sentence_split_docs, 'chroma_sentence_split')

# 3. Token based splitting
# splits text into chunks based on token (words or sub-words) boundaries like tokenizers
# Useful for transformr models that require tokenized input
token_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
token_split_docs = token_splitter.split_documents(docs)
create_vector_db(token_split_docs, 'chroma_token_split')

# 4. Recursive character based splitting
# Attempts to split text at natural boundaries like paragraphs or sections within character limits.
# Balances between chunks coherence and adhering to chunk size
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
recursive_split_docs = recursive_splitter.split_documents(docs)
create_vector_db(recursive_split_docs, 'chroma_recursive_split')

query = "When quit india movement stared?"
query_vector_db(query, 'chroma_char_split')
query_vector_db(query, 'chroma_sentence_split')
query_vector_db(query, 'chroma_token_split')
query_vector_db(query, 'chroma_recursive_split')

