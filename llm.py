import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import tracing

load_dotenv()

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        if not api_key:
            raise ValueError("LLM_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def call_llm(prompt: str, response_model, span_name: str = "llm_call") -> dict:
    client = get_client()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    t0 = time.monotonic()
    response = None
    error_msg = None

    # ⚡ NON-DETERMINISTIC: live API call.
    # This is the only place in the codebase that hits the network and consumes tokens.
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )
        result = response.choices[0].message.parsed.model_dump()
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage if response else None
        tracing.log_span(
            span_name=span_name,
            inputs={"prompt_chars": len(prompt), "model": model, "response_schema": response_model.__name__},
            outputs={"response_chars": len(str(result)) if not error_msg else 0},
            latency_ms=latency_ms,
            metadata={
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            error=error_msg,
        )

    return result
