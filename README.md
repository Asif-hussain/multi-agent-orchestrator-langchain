# Multi-Agent Customer Support System

An intelligent customer support routing system that automatically classifies queries and routes them to specialized RAG agents.

## Overview

This project implements a multi-agent system that:
- Classifies incoming support queries by department (HR, IT, Finance)
- Routes queries to specialized agents with relevant knowledge
- Uses RAG to provide answers based on company documentation
- Tracks everything with Langfuse for observability

## Architecture

```
User Query → Orchestrator (Classification) → Specialized Agent → Response
                                           ├── HR Agent
                                           ├── IT Agent
                                           └── Finance Agent
```

## Project Structure

```
assignment-ai/
├── data/
│   ├── hr_docs/       # HR documentation
│   ├── tech_docs/     # IT documentation
│   └── finance_docs/  # Finance documentation
├── src/agents/
│   ├── hr_agent.py
│   ├── tech_agent.py
│   ├── finance_agent.py
│   ├── orchestrator.py
│   └── evaluator.py
├── multi_agent_system.ipynb
├── test_queries.json
└── requirements.txt
```

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key (or OpenRouter)
- Langfuse account (free at cloud.langfuse.com)

### Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running

Open and run the Jupyter notebook:
```bash
jupyter notebook multi_agent_system.ipynb
```

Execute cells in order to:
1. Load documents and create vector stores
2. Initialize agents
3. Test with sample queries
4. View results in Langfuse

## Usage

```python
from src.agents import OrchestratorAgent, HRAgent, TechAgent, FinanceAgent

# Initialize agents
hr = HRAgent()
hr.initialize()

tech = TechAgent()
tech.initialize()

finance = FinanceAgent()
finance.initialize()

# Create orchestrator
orchestrator = OrchestratorAgent(hr, tech, finance)

# Process query
result = orchestrator.process_query("How many PTO days do I get?")
print(result['answer'])
```

## Testing

The `test_queries.json` file contains 15 test queries covering different departments. Run them all in the notebook to verify the system works correctly.

## Configuration

All system configuration is managed through environment variables in the `.env` file. This allows you to change behavior without modifying code.

### Required Settings

```bash
# OpenRouter API Key
OPENROUTER_API_KEY=your-key-here

# Langfuse Keys (for observability)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Optional Settings

Customize these in `.env` (defaults shown):

```bash
# Models
EMBEDDING_MODEL=openai/text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
EVALUATOR_MODEL=gpt-3.5-turbo

# RAG Parameters
CHUNK_SIZE=1000              # Document chunk size
CHUNK_OVERLAP=200            # Overlap between chunks
RETRIEVAL_K=4                # Number of chunks to retrieve

# Model Behavior
TEMPERATURE=0.1              # Response randomness (0.0-2.0)

# Logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

See `.env.example` for complete documentation with recommendations and cost estimates.

## Technical Decisions

**LangChain**: Used for standardized RAG components and agent orchestration. Makes the code more maintainable and production-ready.

**RAG Approach**: Grounds responses in actual company documentation to avoid hallucinations. Easier to update than fine-tuning models.

**Langfuse**: Provides full observability into the system. Critical for debugging misclassifications and monitoring performance.

**Structured Outputs**: Using Pydantic for classification ensures type-safe routing and consistent responses.

## Bonus Feature

Implemented an Evaluator agent that automatically scores responses on 5 dimensions:
- Relevance
- Completeness
- Accuracy
- Clarity
- Overall quality (1-10 scale)

Scores are logged to Langfuse for tracking quality over time.

## Monitoring

View traces and scores in Langfuse dashboard:
1. Go to cloud.langfuse.com
2. Select your project
3. Check "Traces" tab for execution details
4. Check "Scores" tab for quality metrics

## Known Issues

- Only supports HR, IT, and Finance departments
- English language only
- Queries spanning multiple departments route to single department
- Vector stores need to be rebuilt when documents change

## Dependencies

Main packages:
- langchain
- langchain-openai
- langchain-community
- langchain-chroma
- langfuse
- chromadb
- openai
- python-dotenv
- jupyter

See requirements.txt for complete list with versions.
