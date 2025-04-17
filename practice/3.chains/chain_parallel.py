from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableParallel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
RED = '\033[91m'  # change it, according to the color need
GREEN = '\033[92m'
END = '\033[0m'

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are an tech product reviewer having 20 years of experience in the field"),
        ("human", "List down the all the features of {product} in a single line")
    ]
)
product = input("Enter the product: ")
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

def fetch_pros_features(features):
    pros_features_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("You are an tech product reviewer having 20 years of experience in the field"),
            ("human", "Of these {features}, which are pros comparaed to the similar price ranged competition?")
        ]
    )
    return pros_features_prompt.invoke({'features': features})

def fetch_cons_features(features):
    cons_features_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("You are an tech product reviewer having 20 years of experience in the field"),
            ("human", "Of these {features}, which are cons comparaed to the similar price ranged competition?")
        ]
    )
    return cons_features_prompt.invoke({'features': features})

pros_branch = RunnableLambda(lambda x: fetch_pros_features(x)) | model | StrOutputParser()
cons_branch = RunnableLambda(lambda x: fetch_cons_features(x)) | model | StrOutputParser()

chain = (
    prompt_template | 
    model | 
    StrOutputParser() |
    RunnableParallel({'pros': pros_branch, 'cons': cons_branch}) | 
    RunnableLambda(lambda x: f"\nPros: {GREEN}{x['pros']}{END} \n Cons: {RED}{x['cons']}{END}")   
)  
response = chain.invoke({'product': product})
print(response)
