"""LLMOps routing layer.

Wraps the Azure OpenAI client with logging and a single choke point for model
routing. Passing kwargs through (deployment, temperature, json_mode, system)
lets the eval harness route the same prompt to different deployments to compare
models. Re-raises LLMUnavailable so callers can fall back deterministically.
"""

import logging

from llm.azure_openai_client import run_llm, LLMUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - LLMOps - %(levelname)s - %(message)s",
)

logger = logging.getLogger("llmops.router")


def route_llm(prompt, **kwargs):
    """Route a prompt to the enterprise LLM. Raises LLMUnavailable on failure."""
    deployment = kwargs.get("deployment") or "default"
    try:
        logger.info("Routing request to Azure OpenAI (deployment=%s)", deployment)
        response = run_llm(prompt, **kwargs)
        logger.info("LLM response received (%d chars)", len(response or ""))
        return response
    except LLMUnavailable:
        logger.warning("LLM unavailable for deployment=%s", deployment)
        raise
