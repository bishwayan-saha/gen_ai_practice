import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain import hub
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

current_path = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_path, "db", "chroma_hybrid_search")
file_path = "/home/bishwayansaha99/langchain/docs/rag_nlp.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_chunks = splitter.split_documents(documents)


def load_vector_store_retriever():
    if not os.path.exists(db_path):
        print("===== Creating New Vector Database =====")
        Chroma.from_documents(
            documents=doc_chunks, embedding=embedding, persist_directory=db_path
        )

    vector_score = Chroma(embedding_function=embedding, persist_directory=db_path)
    retriever = vector_score.as_retriever()
    return retriever


def load_keyword_retriever():
    return BM25Retriever.from_documents(documents=doc_chunks)


vector_store_retriever = load_vector_store_retriever()
keyword_retriever = load_keyword_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_store_retriever, keyword_retriever], weights=[0.6, 0.4]
)

prompt_template = hub.pull("rlm/rag-prompt")

print(f"\t\t===== Promt =====\n{prompt_template}\n")


def extract_text_from_doc(documents):
    return "\n\n".join(doc.page_content for doc in documents)


query = input("You: ")

rag_vector_chain = (
    {
        "context": vector_store_retriever | extract_text_from_doc,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | model
    | StrOutputParser()
)

rag_hybrid_chain = (
    {
        "context": ensemble_retriever | extract_text_from_doc,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | model
    | StrOutputParser()
)

print(f"Vector Search Result: \n{rag_vector_chain.invoke(query)}")

print(f"Hybrid Search Result: \n{rag_hybrid_chain.invoke(query)}")

# relevant_docs = vector_store_retriever.invoke(query)
# print(" ########## Vector Search ##########\n\n")
# for doc in relevant_docs:
#     print("================================")
#     print(doc.page_content)
#     print(f"%%% {doc.metadata} %%%")

# relevant_docs = ensemble_retriever.invoke(query)
# print(" ########## Hybrid Search Search ##########\n\n")
# for doc in relevant_docs:
#     print("================================")
#     print(doc.page_content)
#     print(f"%%% {doc.metadata} %%%")

# Creating a compression retriever
compressor = CohereRerank(model = "rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
# relevant_docs = compression_retriever.get_relevant_documents("What is RAG token?")
# print(relevant_docs)

rag_rerank_chain = (
    {
        "context": compression_retriever | extract_text_from_doc,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | model
    | StrOutputParser()
)

print(f"Hybrid Reranked Search Result: \n{rag_rerank_chain.invoke(query)}")