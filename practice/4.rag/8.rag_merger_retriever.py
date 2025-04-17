import os
from re import search

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain import hub
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

load_dotenv()
gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
cohere_embedding = CohereEmbeddings(model="embed-english-v3.0")
llama = OllamaLLM(model="llama3.2")

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path_1 = "/home/bishwayansaha99/langchain/docs/attention.pdf"
file_path_2 = "/home/bishwayansaha99/langchain/docs/lost_in_the_middle.pdf"

db_path_1 = os.path.join(current_dir, "db", "chroma_merger_retriever_1")
db_path_2 = os.path.join(current_dir, "db", "chroma_merger_retriever_2")


def load_vector_db(file_path: str, db_path: str):
    collection_name = file_path[file_path.index("/docs") + 6 : file_path.index(".pdf")]
    if not os.path.exists(db_path):

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunk_docs = splitter.split_documents(documents)

        print(collection_name)
        Chroma.from_documents(
            documents=chunk_docs,
            embedding=gemini_embedding,
            persist_directory=db_path,
            collection_name=collection_name,
        )
    vector_store = Chroma(
        embedding_function=gemini_embedding,
        persist_directory=db_path,
        collection_name=collection_name,
    )
    retriever = vector_store.as_retriever()
    return retriever


retriever_1 = load_vector_db(file_path=file_path_1, db_path=db_path_1)
retriever_2 = load_vector_db(file_path=file_path_2, db_path=db_path_2)

merger_retriever = MergerRetriever(retrievers=[retriever_1, retriever_2])

# print(merger_retriever.invoke("What is attention model?"))
filter = EmbeddingsRedundantFilter(embeddings=cohere_embedding)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])

# Contextual compression compresses base retriver retrived documents using the context of given user query so that only
# relevant information is passed to the LLM for better result.
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=merger_retriever,
    search_kwargs={"k": 3, "include_metadata": 3},
)

prompt_template = hub.pull("rlm/rag-prompt")


def extract_text_from_doc(documents):
    print([doc.metadata for doc in documents])
    return "\n\n".join(doc.page_content for doc in documents)


query = input("You: ")

rag_merger_chain = (
    {
        "context": compression_retriever | extract_text_from_doc,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | llama
    | StrOutputParser()
)

print(rag_merger_chain.invoke(query))
