import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")

print("Groq API key configured.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

try:
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
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
    usage = response.usage
    print(f"Prompt Tokens: {usage.prompt_tokens if usage else 0}")
    print(f"Response Tokens: {usage.completion_tokens if usage else 0}")
except OpenAIError as exc:
    print(f"API request failed: {exc}")
    print("Prompt Tokens: 0")
    print("Response Tokens: 0")
