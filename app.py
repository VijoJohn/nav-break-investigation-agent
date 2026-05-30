"""NAV Break Investigation — demo entry point.

Detect NAV breaks from the sample datasets, then run each break through the
wired LangGraph investigation agent (chunked RAG retrieval -> classify & cite
-> reflect). Runs end to end with or without an Azure OpenAI deployment: with
a key it uses the LLM; without one it uses the transparent offline fallback.
"""

import os

from dotenv import load_dotenv

from controls.break_detection import detect_breaks
from llm.azure_openai_client import is_configured
from orchestration.agent_workflow import investigate

load_dotenv()


def main():
    print("\nAGENTIC AI - NAV BREAK INVESTIGATION PROTOTYPE\n")

    mode = "Azure OpenAI" if is_configured() else "offline fallback"
    print(f"LLM mode: {mode}\n")

    breaks = detect_breaks()

    if not breaks:
        print("No NAV breaks detected. NAV validation checks passed.")
        return

    print(f"Detected {len(breaks)} NAV break(s).\n")

    for i, brk in enumerate(breaks, start=1):
        result = investigate(brk["observation"])

        print("=" * 70)
        print(f"BREAK {i}: {brk['subject']}")
        print(f"  Detected category : {brk['category']}")
        print(f"  Agent category    : {result['category']}")
        print(f"  Cited rules       : {', '.join(result['cited_rules']) or '-'}")
        print(f"  Root cause        : {result['root_cause']}")
        print(f"  Recommended control: {result['recommended_control']}")
        if result["investigation_steps"]:
            print("  Investigation steps:")
            for step in result["investigation_steps"]:
                print(f"    - {step}")
        print(f"  Summary           : {result['summary']}")
        print(
            f"  (agent backend: {result['backend']}, "
            f"retrieval: {result['retrieval_backend']})"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
