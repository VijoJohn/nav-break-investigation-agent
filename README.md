# Agentic AI for NAV Break Investigation

This project demonstrates how **Agentic AI architectures can support NAV break investigations in fund accounting operations**.

The prototype combines **fund accounting control logic with AI investigation workflows** to simulate how operations teams analyze unexpected NAV movements across pricing, positions, and accruals.

---

# Business Context

Daily NAV validation is one of the most critical control processes in fund accounting.

Unexpected NAV movements may arise from several operational drivers including:

- Security price movements
- Income accrual differences
- Expense accrual discrepancies
- Distribution adjustments
- Position changes

When a NAV break occurs, operations teams must quickly identify the underlying driver before confirming the final NAV.

This prototype explores how **Agentic AI workflows can assist in that investigation process.**

---

# Architecture Overview

The system combines traditional financial controls with modern AI infrastructure.

### Fund Accounting Control Layer

Operational control checks simulate common NAV validation procedures:

- Income accrual validation
- Expense accrual validation
- Unrealized gain/loss analysis
- Distribution verification

These checks represent the **first layer of NAV investigation**.

---

### Break Detection Agent

The break detection logic identifies potential anomalies across:

- security pricing
- accrual calculations
- distribution entries

When discrepancies exceed tolerance thresholds, an AI investigation workflow is triggered.

---

### Evolved RAG Knowledge Layer

The system retrieves contextual knowledge using **Retrieval-Augmented Generation (RAG)**.

Knowledge sources include:

- NAV validation rules
- accounting procedures
- investigation playbooks

Technologies used:

- Qdrant Vector Database
- LangGraph orchestration

---

### Agentic Memory

Historical NAV investigations are stored in a memory layer including:

- prior break patterns
- root causes
- resolution steps
- recommended controls

This allows the system to reference previous investigations when similar patterns occur.

---

### AI Investigation Workflow

An investigation agent analyzes the detected break using enterprise language models.

Enterprise model layer:

Azure OpenAI

The AI generates investigation insights describing:

- likely root causes
- investigation steps
- suggested operational controls

---

### Reflection Agent

A reflection agent reviews and refines the investigation output to improve reasoning clarity and accuracy.

---

### LLMOps Layer

The architecture includes an LLMOps layer responsible for:

- prompt routing
- monitoring and logging
- evaluation and testing
- model governance

This ensures the system remains **observable, controllable, and production-ready.**

---

# Repository Structure

nav-break-investigation-agent

data/
positions.csv
prices.csv
income_accruals.csv
expense_accruals.csv
distributions.csv

knowledge/
nav_validation_rules.txt
investigation_playbooks.txt

memory/
nav_break_memory.json

llm/
azure_openai_client.py

llmops/
llm_router.py
evaluation.py

app.py
requirements.txt
README.md
.env.example
---

# Technology Stack

Python  
Azure OpenAI  
LangGraph  
Qdrant Vector Database  
Sentence Transformers  
Pandas

---

# Running the Prototype

Install dependencies:
pip install -r requirements.txt

Run the prototype:
python app.py

---

# Environment Configuration

Create your environment configuration file:

Add your Azure OpenAI credentials:

---

# Example Output
NAV BREAK INVESTIGATION SUMMARY

Driver
XYZ Equity price increased 15% vs prior day.

Historical Pattern
Similar break occurred previously due to stale pricing.

Recommended Investigation
Verify pricing vendor feed.

Suggested Control
Implement automated price tolerance validation.

---

# Purpose of the Prototype

This project demonstrates how **Agentic AI architectures can augment operational workflows in financial services**, particularly in areas such as:

- NAV validation
- break investigation
- financial reporting controls
- operational risk monitoring

The objective is not to replace fund accounting controls, but to **enhance investigation workflows with intelligent automation.**

---

# Future Enhancements

Potential extensions include:

- integration with pricing vendor feeds
- anomaly detection models
- multi-agent orchestration using LangGraph
- automated reporting validation
