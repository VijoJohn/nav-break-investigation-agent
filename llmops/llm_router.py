import logging
from llm.azure_openai_client import run_llm


logging.basicConfig(level=logging.INFO)


def route_llm(prompt):

    logging.info("LLMOps Router: sending prompt to enterprise model")

    response = run_llm(prompt)

    logging.info("LLMOps Router: response received")

    return response
