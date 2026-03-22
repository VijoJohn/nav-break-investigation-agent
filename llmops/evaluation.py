import logging


logging.basicConfig(level=logging.INFO)


def evaluate_investigation(response):

    """
    Simple evaluation layer for LLM responses.
    In production environments this would include
    automated tests, scoring, and monitoring.
    """

    logging.info("Running LLM response evaluation")

    if len(response) < 50:
        logging.warning("Response may be too short")

    if "NAV" not in response:
        logging.warning("Response may lack NAV context")

    logging.info("Evaluation completed")

    return True
