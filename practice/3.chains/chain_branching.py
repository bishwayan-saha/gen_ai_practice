from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You understand human psycology and sentimnets through their statements."),
        ("human", "Categorized the following feedback {feedback} as positive, negative and neutral and return the response as one of the options.")
    ]
)

feedback = "I really love the product. My life got easy after using this product"
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

positive_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You understand human psycology and sentimnets through their statements."),
        ("human", "Provide a short thank you message for the positive feedback: {feedback}")
    ]
)


negative_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You understand human psycology and sentimnets through their statements."),
        ("human", "Provide a short sorry message for the negative feedback: {feedback}")
    ]
)


neutral_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You understand human psycology and sentimnets through their statements."),
        ("human", "Ask for more details for the neutral feedback: {feedback}")
    ]
)

branch_chain = RunnableBranch(
    (
        lambda x: 'positive' in x,
        positive_prompt | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x,
        negative_prompt | model | StrOutputParser()
    ),

    neutral_prompt | model | StrOutputParser()
    
)

classification_chain = prompt_template | model | StrOutputParser()
chain = classification_chain | branch_chain
response = chain.invoke({'feedback': feedback})
print(response)
