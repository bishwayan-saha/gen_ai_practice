from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_ollama import OllamaLLM
import datetime 
from wikipedia import summary

load_dotenv()

def get_current_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_wikipedia_summary(person):
    return summary(person, sentence = 3)

@tool
def get_employee_id_by_name(name: str):
    """
        Get employee id given the employee name
    """
    fake_employee_data = {
        "Alex": "E001",
        "Ben": "E002",
        "Clarke": "E003",
        "Dean": "E004",
        "Emily": "E005",
        "Frank": "E006",
        "Gus": "E007",
        "Henry": "E008"
    }
    return fake_employee_data.get(name, "Employee name does't exist")

@tool
def get_employee_sal_by_id(emp_id: str):
     """
        Get employee salary given the employee id
    """
     fake_employee_data = {
         "E001": 23000,
         "E002": 34000,
         "E003": 27000,
         'E004': 28000,
         "E005": 39000,
         "E006": 45000,
         "E007": 21000,
         "E008": 36000
     }
     return fake_employee_data.get(emp_id, "Employee id not found")

tools = [
    Tool(
        name = "current_time",
        func=get_current_time,
        description= "Get the current time in the format: %Y-%m-%d %H:%M:%S"
    ),
    Tool(
        name = "wikipedia_summary",
        func=get_wikipedia_summary,
        description= "Get a summary of a person from Wikipedia"
    ),
    Tool(
        name = "get_employee_id",
        func=get_employee_id_by_name,
        description="Given a name, it will return the employee id"
    ),
    Tool(
        name = "get_employee_salary",
        func=get_employee_sal_by_id,
        description="Given a employee id, it will return the employee salary"
    )
]

# prompt = hub.pull("hwchase17/structured-chat-agent")
prompt = hub.pull("hwchase17/react")

# llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
llm = OllamaLLM(model="llama3.2")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

memory.chat_memory.add_messages(("system", "You are an AI assistant, who uses available tools to answer questions. If you don't know the answer, reply with 'I don't know'."))

print(" ===== Starting conversation ===== \n")
print(" To stop the conversation, type 'exit' \n")
while True:
    user_input = input("You: ")
    if user_input.lower().strip() == 'exit':
        break
    memory.chat_memory.add_messages(("human", user_input))
    response = agent_executor.invoke({"input": user_input})
    memory.chat_memory.add_messages(("ai", response["output"]))
    print(f"AI: {response['output']}")
    print(response)


