from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    api_key=SecretStr(api_key)
)

response = llm.invoke("Hi, this is a test message to check API availability.")
print("API Response:", response) 