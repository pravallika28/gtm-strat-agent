import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        # DETERMINISTIC: just instantiates the SDK client, no API call yet
        _client = genai.Client(api_key=api_key)
    return _client


def call_gemini(prompt: str, response_model) -> dict:
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ⚡ NON-DETERMINISTIC: live API call to Gemini.
    # Output varies across runs even with identical input — temperature > 0 by default.
    # This is the only place in the codebase that hits the network and consumes tokens.
    # All non-determinism in the system flows through here.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_model,
        ),
    )
    # DETERMINISTIC: JSON parsing of the response text
    return json.loads(response.text)
