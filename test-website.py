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
        task="""Test the responsive design of https://letstalkaitools.com across various device types and screen sizes.

1. Test the following screen sizes and orientations:
   - Mobile: 320px, 375px, 414px (portrait and landscape)
   - Tablet: 768px, 1024px (portrait and landscape)
   - Desktop: 1366px, 1920px

2. For each combination, evaluate:
   - Content visibility and readability
   - Navigation usability
   - Image and media scaling
   - Form functionality
   - Touch targets and interactive elements
   - Load time and performance

3. Test specific responsive features:
   - Hamburger menu functionality on mobile
   - Image resizing (or disappearing) on mobile (to never block any text)

Document all issues with screenshots, device information, and recommended fixes. Prioritize issues based on severity and frequency of user encounter.""",
        llm=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            api_key=SecretStr(api_key)
        ),
    )
    await agent.run()

asyncio.run(main()) 