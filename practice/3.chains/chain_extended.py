from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a history professor who is expert in European history and exlains everything in short and simple words"),
        ("human", "What are 3 major incidents, happened in {year}?")
    ]
)
year = input("Enter the year: ")
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
uppercase = RunnableLambda(lambda x: x.upper())
word_count = RunnableLambda(lambda x: f'{len(x.split())} words. \n {x}')
chain = prompt_template | model | StrOutputParser() | uppercase | word_count
response = chain.invoke({'year': year})
print(response)
