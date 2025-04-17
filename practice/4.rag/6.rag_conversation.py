from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
import os

GREEN = "\033[92m"
RED = "\033[91m"
END = "\033[0m"

current_dir = os.path.dirname(os.path.abspath(__file__))
persistant_dir = os.path.join(current_dir, "db", "chroma_gemini_embedding")

gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

db = Chroma(persist_directory=persistant_dir, embedding_function=gemini_embedding)
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.5},
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

history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """   
                You are an question answer based assistant. Please use the following piece of retrieved information from
                      vector store to answer the question. If you don't know the answer, reply with "I don't know".
                      Try to answer within 3 sentence with easy to understand language and keep the answer to the point.
                      \n\n
                      {context}
                      """,
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def conversation():
    print(" ===== Starting conversation ===== \n")
    print(" To stop the conversation, type 'exit' \n")
    chat_history = []
    while True:
        query = input("You:")
        if query.lower().strip() == "exit":
            break
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})
        print(response)
        print(f"AI: {response["answer"]}")
        chat_history.append(HumanMessage(query))
        chat_history.append(AIMessage(response["answer"]))


if __name__ == "__main__":
    conversation()
