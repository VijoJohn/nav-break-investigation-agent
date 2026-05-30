"""Parse the knowledge base into retrievable, rule-tagged chunks.

Rather than embedding whole files as a single vector (which makes retrieval
meaningless), this splits the validation rules and the investigation playbook
into individual chunks, each tagged with the rule id it belongs to ([R1]..[R6]).
The vector store then indexes one chunk per point so retrieval can return the
specific rule and playbook step relevant to a break.
"""

import os
import re

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")

RULES_FILE = os.path.join(KNOWLEDGE_DIR, "nav_validation_rules.txt")
PLAYBOOK_FILE = os.path.join(KNOWLEDGE_DIR, "investigation_playbooks.txt")

_RULE_TAG = re.compile(r"\[(R\d+)\]")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_chunks():
    """Return a list of knowledge chunks.

    Each chunk is a dict: {"rule_id": str|"", "source": str, "text": str}.
    Lines tagged with [Rn] become individual rule/playbook chunks; the playbook
    is split on its tagged driver headings so each driver and its steps stay
    together.
    """
    chunks = []

    # --- Validation rules: one chunk per tagged line ---
    for line in _read(RULES_FILE).splitlines():
        line = line.strip()
        if not line:
            continue
        m = _RULE_TAG.match(line)
        if m:
            chunks.append(
                {
                    "rule_id": m.group(1),
                    "source": "nav_validation_rules",
                    "text": line,
                }
            )

    # --- Playbook: split into driver blocks keyed by their [Rn] heading ---
    playbook = _read(PLAYBOOK_FILE)
    # Split on tagged headings while keeping the tag with its block.
    blocks = re.split(r"\n(?=\[R\d+\])", playbook)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = _RULE_TAG.search(block)
        if m:
            chunks.append(
                {
                    "rule_id": m.group(1),
                    "source": "investigation_playbook",
                    "text": block,
                }
            )

    return chunks


if __name__ == "__main__":
    for c in load_chunks():
        print(f"{c['rule_id']:>4} [{c['source']}] {c['text'][:70]}...")
