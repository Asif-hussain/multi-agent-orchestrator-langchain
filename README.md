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

Key parameters in agent files:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Top-k retrieval: 4 documents
- Temperature: 0.0 for classification, 0.1 for generation

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
