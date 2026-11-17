
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY missing. Check your .env file."
    )

print("API key detected successfully!")

# Connect to Groq
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0,
    groq_api_key=api_key
)

print("Connecting to Groq...")

# Send a test request
response = llm.invoke(
    "Reply with exactly: Groq connection successful!"
)

print("\nGroq Response:")
print(response.content)

print("\nAPI CONNECTION TEST PASSED!")
