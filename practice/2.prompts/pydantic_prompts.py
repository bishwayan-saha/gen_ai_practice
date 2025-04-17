from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from enum import Enum
from langchain_ollama import OllamaLLM

class Period(int, Enum):
    six = 6
    twelve = 12
    eighteen = 18
    twenty_four = 24
    thirty_six = 36

class Product(BaseModel):
    product_name: Optional[str] = Field(..., description="Name of the product")
    processor: Optional[str] = Field(..., description="Name of the processor used in the product")
    gpu: Optional[str] = Field(..., description="Name of the GPU used in the product")
    display: Optional[str] = Field(..., description="Display configuration (resolution, refresh rate, LCD or LED etc.) of the product")
    keyboard: Optional[str] = Field(..., description="Keyboard layout used in the product")
    price: Optional[int] = Field(..., description="Price of the product in rupees.")
    warranty: Optional[Period] = Field(..., description="Warranty period of the product in months.")
load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("""
        You have the technical knowledge of a product and you have to provide the details of the product.
        Only extract relevant information from the text.
        If you do not have the information, you can skip the field and pass null in the respective field.
                      """),
        ("human", "What is the complete specificatio details of {product}")
    ]
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
# model = OllamaLLM(model="llama3.2")
structured_model = model.with_structured_output(Product)

product = input("Enter the product name: ")
prompt = prompt_template.invoke({'product': product})
response: Product = structured_model.invoke(prompt)
print(response)