import logging
from llm.azure_openai_client import run_llm


# ---------------------------------------
# LLMOps Configuration
# ---------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - LLMOps - %(levelname)s - %(message)s"
)


# ---------------------------------------
# Router Function
# ---------------------------------------

def route_llm(prompt):

    try:

        logging.info("Routing request to enterprise LLM (Azure OpenAI)")

        response = run_llm(prompt)

        logging.info("LLM response received successfully")

        return response

    except Exception as e:

        logging.error("LLM call failed")

        return f"LLM investigation failed due to: {str(e)}"
