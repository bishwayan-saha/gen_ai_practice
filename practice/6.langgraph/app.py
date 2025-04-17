from langchain_ollama import OllamaLLM
from transformers import CLIPModel

model = OllamaLLM(model="llama3.2")
embedding = CLIPModel.from_pretrained(model_name = "openai/clip-vit-base-patch16")