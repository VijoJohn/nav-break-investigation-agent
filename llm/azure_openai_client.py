"""Azure OpenAI chat client.

Thin wrapper around the Azure OpenAI Chat Completions API. Supports:
  - both AZURE_OPENAI_API_KEY and AZURE_OPENAI_KEY env var names,
  - per-call deployment override (used by the eval harness to compare models),
  - optional JSON response mode for structured agent output,
  - a custom system prompt.

Raises LLMUnavailable when no key/endpoint is configured or the call fails, so
callers can fall back to the offline path deterministically.
"""

import os


class LLMUnavailable(Exception):
    """Raised when the Azure OpenAI deployment is not configured or unreachable."""


def _api_key():
    return os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")


def is_configured():
    return bool(_api_key() and os.getenv("AZURE_OPENAI_ENDPOINT"))


def run_llm(prompt, deployment=None, temperature=0.2, json_mode=False, system=None):
    """Call Azure OpenAI and return the message content as a string.

    Args:
        prompt: user message content.
        deployment: deployment/model name; defaults to AZURE_OPENAI_DEPLOYMENT.
        temperature: sampling temperature.
        json_mode: if True, request a JSON object response.
        system: optional system prompt override.
    """
    if not is_configured():
        raise LLMUnavailable("Azure OpenAI not configured (missing key or endpoint)")

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=_api_key(),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        kwargs = {
            "model": deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            "messages": [
                {
                    "role": "system",
                    "content": system or "You are a fund accounting investigation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    except LLMUnavailable:
        raise
    except Exception as exc:
        raise LLMUnavailable(f"Azure OpenAI call failed: {exc}") from exc
