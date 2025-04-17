import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


def get_response_from_llm(user_question: str, chat_history: list):
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an AI assistant. By considering the previous chat history and user question, return the response.

        Chat History: {chat_history} \n
        User Question: {user_question}
"""
    )

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    chain = prompt_template | model | StrOutputParser()

    return chain.stream({"user_question": user_question, "chat_history": chat_history})


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage("Hi I am your AI streaming assistant")]

st.set_page_config(page_title="Streaming Bot", page_icon="🌊")
st.title("Streaming Bot")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

user_input = st.chat_input("Type your message...")

if user_input is not None and user_input.strip() != "":
    st.session_state.chat_history.append(HumanMessage(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("ai"):
        response = st.write_stream(get_response_from_llm(user_question=user_input, chat_history=st.session_state.chat_history)) 
    st.session_state.chat_history.append(AIMessage(response))
