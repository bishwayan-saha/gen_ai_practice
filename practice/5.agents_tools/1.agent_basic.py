from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime 

load_dotenv()

def get_current_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(
        name = "current_time",
        func=get_current_time,
        description= "Get the current time in the format: %Y-%m-%d %H:%M:%S"
    ),
]

prompt = hub.pull("hwchase17/react")

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

agent = create_react_agent(llm=llm, prompt=prompt, tools=tools, stop_sequence=True)

agent_executor = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True)

response = agent_executor.invoke({"input": "What is the current time?"})

print(response)


