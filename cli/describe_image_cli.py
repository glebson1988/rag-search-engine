#!/usr/bin/env python3

import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe image and rewrite query")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--query", type=str, required=True, help="Text query")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")

    # Keep this wording close to assignment requirements.
    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve "
        "search results from a movie database. Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )

    if not api_key:
        print(f"Rewritten query: {args.query.strip()}")
        print("Total tokens:    0")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    img_b64 = base64.b64encode(img).decode("ascii")

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.query.strip()},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                    ],
                },
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip() or args.query.strip()
        print(f"Rewritten query: {rewritten}")

        total_tokens = 0
        if response.usage is not None and getattr(response.usage, "total_tokens", None) is not None:
            total_tokens = response.usage.total_tokens
        print(f"Total tokens:    {total_tokens}")
    except OpenAIError:
        print(f"Rewritten query: {args.query.strip()}")
        print("Total tokens:    0")


if __name__ == "__main__":
    main()
