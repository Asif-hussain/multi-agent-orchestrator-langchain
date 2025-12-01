#!/usr/bin/env python3
"""
Quick test to verify evaluator scoring is working with Langfuse
"""

import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from src.agents import HRAgent, TechAgent, FinanceAgent, OrchestratorAgent, EvaluatorAgent

# Load environment
load_dotenv()

print("="*80)
print("EVALUATOR SCORING TEST WITH TRACE ID CAPTURE")
print("="*80)

# Initialize Langfuse
print("\n1. Initializing Langfuse...")
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
langfuse_handler = CallbackHandler()
print("    Langfuse initialized")

# Initialize agents (minimal setup)
print("\n2. Initializing agents...")
hr_agent = HRAgent()
hr_agent.initialize('data/hr_docs')

tech_agent = TechAgent()
tech_agent.initialize('data/tech_docs')

finance_agent = FinanceAgent()
finance_agent.initialize('data/finance_docs')

# Pass langfuse_handler to orchestrator
orchestrator = OrchestratorAgent(hr_agent, tech_agent, finance_agent, langfuse_handler=langfuse_handler)
evaluator = EvaluatorAgent(langfuse_client=langfuse)
print("    All agents initialized with Langfuse handler")

# Test with a simple query
print("\n3. Processing test query...")
print("="*80)
query = "How many PTO days do I get?"
result = orchestrator.process_query(query, verbose=True)

# Get trace_id from result
trace_id = result.get('trace_id')
print(f"\n   Captured trace_id: {trace_id[:32] if trace_id else 'None'}...")

print("\n4. Evaluating response with trace_id...")
print("="*80)
evaluation = evaluator.evaluate_response(
    query=result["query"],
    answer=result["answer"],
    department=result["classification"]["department"],
    source_documents=result["source_documents"],
    trace_id=trace_id  # IMPORTANT: Pass trace_id to link scores!
)

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Overall Score: {evaluation.overall_score}/10")
print(f"Relevance: {evaluation.relevance_score}/10")
print(f"Completeness: {evaluation.completeness_score}/10")
print(f"Accuracy: {evaluation.accuracy_score}/10")
print(f"Clarity: {evaluation.clarity_score}/10")

# Ensure data is sent
print("\n5. Flushing data to Langfuse...")
langfuse.flush()
print("    Data flushed")

print("\n" + "="*80)
print(" TEST COMPLETE!")
print("="*80)
if trace_id:
    print("\n SUCCESS! Trace ID was captured and scores were logged!")
    print(f"   Trace ID: {trace_id}")
    print("\nNext steps:")
    print("1. Go to: https://cloud.langfuse.com")
    print("2. Navigate to your project")
    print("3. Click on 'Scores' tab")
    print("4. You should see 5 score entries linked to the trace:")
    print("   - overall_quality")
    print("   - relevance")
    print("   - completeness")
    print("   - accuracy")
    print("   - clarity")
else:
    print("\n WARNING: No trace_id was captured!")
    print("   Scores were calculated but not linked to Langfuse trace")
print("="*80)
