from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

chat_history = []
system_msg = SystemMessage("You are smart AI assitant that answers in very brief way without spending much words and time")

print("Type 'exit' to end the conversation")
name = input("What is your name? ")

while True:
    query = input(f"{name}: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(query))
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
    response = model.invoke([system_msg] + chat_history).content
    print(f"AI: {response}")
    chat_history.append(AIMessage(response))