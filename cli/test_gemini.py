import os

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")

print(f"Using key {api_key[:6]}...")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": (
                    "Why is Boot.dev such a great place to learn about RAG? "
                    "Use one paragraph maximum."
                ),
            }
        ],
    )
    text = response.choices[0].message.content or ""
    print(text)
    print(f"Prompt Tokens: {response.usage.prompt_tokens}")
    print(f"Response Tokens: {response.usage.completion_tokens}")
except OpenAIError as exc:
    print(f"API request failed: {exc}")
    print("Prompt Tokens: 0")
    print("Response Tokens: 0")
