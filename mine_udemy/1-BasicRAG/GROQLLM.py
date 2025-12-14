from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")
result = llm.invoke("how are you ?")
print(result)
