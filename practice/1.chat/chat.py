from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# Example usage
load_dotenv()
messages = [
    # System Message shoul always be the first message
    SystemMessage("Youa are a PhD. in mathematics"),
    HumanMessage("What is divison result of 81 by 9?")
]

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

response = model.invoke(messages)
print(response)

