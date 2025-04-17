import ctransformers.transformers
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def get_llama_response(input_text, no_of_words, language_type):
    llm = CTransformers(model = "C:/Users/bishw/Documents/models/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type = 'llama',
                        config = {"max_new_tokens": 256, "temperature": 0.1})
    
    template = """Write a blog on the topic '{input_text}' in {language_type} style within {no_of_words} words"""
    prompt = PromptTemplate(input_variables=["input_text", "language_type", "no_of_words"], template=template)
    response = llm(prompt.format(input_text = input_text, language_type = language_type, no_of_words = no_of_words))
    print(response)
    return response


st.set_page_config(
    page_title="Generate Blogs",
    page_icon="XXX",
    layout="centered",
    initial_sidebar_state="collapsed"
    )

st.header("Generate Blogs")


input_text = st.text_input("Enter the blog topic")
col1, col2 = st.columns([5,5])

with col1:
    no_of_words = st.text_input("Enter no of words")

with col2:
    language_type = st.selectbox("How you want the blog sound", 
                                 ("Standard", "Fluent", "Research", "Casual", "Business", "Editorial"),
                                 index=0
                                 )
submit = st.button("Generate")

if submit:
    st.write(get_llama_response(input_text, no_of_words, language_type))

