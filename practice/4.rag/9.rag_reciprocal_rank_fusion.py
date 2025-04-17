import os
from re import search

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import loads, dumps
from pprint import pprint

load_dotenv()

gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llama = OllamaLLM(model="llama3.2")

current_path = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_path, "db", "chroma_rrf")
file_path = "/home/bishwayansaha99/langchain/docs/battle_of_plassey.pdf"


def load_vector_store_retriever():
    if not os.path.exists(db_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        doc_chunks = splitter.split_documents(documents)
        print("===== Creating New Vector Database =====")
        Chroma.from_documents(
            documents=doc_chunks, embedding=gemini_embedding, persist_directory=db_path
        )

    vector_score = Chroma(
        embedding_function=gemini_embedding, persist_directory=db_path
    )
    retriever = vector_score.as_retriever()
    return retriever


vector_retriever = load_vector_store_retriever()


def generate_similar_questions(query: str):

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are an intelligent assistant that generates 5 similar queries based on a given user query. 
                    Here is the question: {question} 
                    The response should have the original query along with the similar query results separated by new line characater '\n' 
                    While returning the response, do NOT generate any redundant prefix, suffix or alphanumeric character or phrases
                    like "Here are the similar queries" etc. Just provide the similar questions only separated by newline.
                """,
            )
        ]
    )

    chain = prompt_template | llama | StrOutputParser() | (lambda x: pprint(x) or x.split("\n"))
    return chain


query = input("You: ")


def reciprocal_rank_function(documents: list[list], k=60):

    fused_scores = {}
    for docs in documents:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


rag_fusion_chain = (
    generate_similar_questions | vector_retriever.map() | reciprocal_rank_function
)

prompt_template = hub.pull("rlm/rag-prompt")

final_chain = (
    {"context": rag_fusion_chain, "question": RunnablePassthrough()}
    | prompt_template
    | llama
    | StrOutputParser()
)

print(final_chain.invoke(query))
