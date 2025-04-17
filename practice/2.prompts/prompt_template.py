from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
COLOR = '\033[91m'  # change it, according to the color need

END = '\033[0m'

prompt_template_1 = ChatPromptTemplate.from_template("Who is {name}?")

prompt = prompt_template_1.invoke({"name": "Albert Einstein"})
print(prompt, type(prompt))
print("---------------------------------------")
prompt_template_2 = ChatPromptTemplate.from_messages(
    [ 
        ("system", "You are a history professor"),
        ("human", "What is the capital of {country}?")
    ]
)

prompt = prompt_template_2.invoke({'country': 'France'})
print(prompt)
print("---------------------------------------")

## This will not work
prompt_template_3 = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a history professor"),
        HumanMessage("What is the capital of {country}?")
    ]
)

prompt = prompt_template_3.invoke({'country': 'France'})
print(prompt)
print(f'{COLOR} This does not work, for string manipulation tuple should be used{END}')
print("---------------------------------------")

prompt_template_4 = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a history professor who is expert in European history and exlains everything in very brief way"),
        ("human", "What major incidents happened in {year}?")
    ]
)
print('---------------------------------------')
year = input("Enter the year: ")
final_prompt = prompt_template_4.invoke({'year': year})
print(final_prompt)

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
response = model.invoke(final_prompt).content
print(response)


