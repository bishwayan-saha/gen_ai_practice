import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os
import datetime
import hashlib

load_dotenv()

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_dir = os.path.join(current_dir, "chroma")

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def create_hash(s: str):
    s = s.encode("utf-8")
    return hashlib.sha224(s).hexdigest()


def get_final_response(user_question, chat_history):
    return "I don't know"


def create_vector_store_from_url(url: str):
    url_hash = create_hash(url)
    db_path = os.path.join(persistent_dir, f"{url_hash}")
    if os.path.exists(db_path):
        print("===== Loading vector store =====")
        return Chroma(persist_directory=db_path, embedding_function=embedding)

    print(
        f"===== Creating vector store at {datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")} ====="
    )

    # Get the text in document format
    loader = WebBaseLoader(url)
    documents = loader.load()

    # split the document into chunks
    splitter = RecursiveCharacterTextSplitter()
    doc_chunks = splitter.split_documents(documents)

    # create a vectorstore
    vector_store = Chroma.from_documents(
        documents=doc_chunks, embedding=embedding, persist_directory=db_path
    )

    return vector_store


def get_context_retrieval_chain(vector_store: Chroma):

    retriever = vector_store.as_retriever(
        # search_type="similarity_score_threshold",
        # search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """
                        Given a chat history and the latest user question which might refrerence context in chat history,
                        reformulate a standalone question that can be understood without the chat history. 
                        \nDo not answer the question, just reformulate it if needed, otherwise return as it is.
                      """
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_reriever = create_history_aware_retriever(
        llm=model, retriever=retriever, prompt=contextualize_q_prompt
    )

    return history_aware_reriever


def get_conversational_chain(retreival_chain):

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are an question answer based assistant. Please use the following piece of retrieved information context from
                    vector store to answer the question. If you don't know the answer, reply with "I don't know".
                    Try to answer within few sentences with easy to understand language.
                    Here is the context. \n
                    {context}
                """,
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_document_chain = create_stuff_documents_chain(
        llm=model, prompt=prompt_template
    )

    return create_retrieval_chain(retreival_chain, stuff_document_chain)


st.set_page_config(page_title="Chat with website", page_icon="🌎")
st.title("Chat with website")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            "Hi I am your web assistant. You type your question and I will try to find the answer from the given webpage url"
        )
    ]

if "disabled" not in st.session_state:
    st.session_state.disabled = True

with st.sidebar:
    st.header("Settings")
    st.text_input(label="URL", key="url")
    if st.button(label="Connect"):
        if st.session_state["url"] is None or st.session_state["url"].strip() == "":
            st.error("Please enter a valid URL")
        else:
            st.success("URL found")
            st.session_state.disabled = False


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("HUMAN"):
            st.markdown(message.content)

user_question = st.chat_input(
    placeholder="Type your message here...", disabled=st.session_state.disabled
)

if user_question is not None and user_question.strip() != "":
    st.session_state.chat_history.append(HumanMessage(user_question))
    with st.chat_message("HUMAN"):
        st.markdown(user_question)
    with st.chat_message("AI"):
        # if "vector_store" not in st.session_state:
        #     st.session_state.vector_store = create_vector_store_from_url(
        #         st.session_state["url"]
        #     )
        # print(f'CHAT HISTORY: {st.session_state.chat_history}')
        # print(f'VECTOR STORE: {st.session_state.vector_store}')
        vector_store = create_vector_store_from_url(st.session_state["url"])
        context_retrieval_chain = get_context_retrieval_chain(vector_store)
        conversational_rag_chain = get_conversational_chain(context_retrieval_chain)
        response = conversational_rag_chain.invoke(
            {"chat_history": st.session_state.chat_history, "input": user_question}
        )
        st.markdown(response["answer"])
    st.session_state.chat_history.append(AIMessage(response["answer"]))
