import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client with your API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Make a simple API call
try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": "Hello! Can you tell me a quick fact about the moon?"}
        ],
        max_tokens=100
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")