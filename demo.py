from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
import os
from pydantic import SecretStr
load_dotenv()

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=SecretStr(api_key)
        ),
    )
    await agent.run()

asyncio.run(main()) 