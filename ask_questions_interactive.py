#!/usr/bin/env python3
"""
Interactive Question-Asking Script
Run this to ask questions and see results + Langfuse traces in real-time
"""

from dotenv import load_dotenv
load_dotenv()

from src.agents import HRAgent, TechAgent, FinanceAgent, OrchestratorAgent, EvaluatorAgent
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import os

def initialize_system():
    """Initialize all agents and return orchestrator and evaluator"""
    print("="*80)
    print("🤖 MULTI-AGENT CUSTOMER SUPPORT SYSTEM")
    print("="*80)
    print()

    # Initialize Langfuse
    print("📊 Initializing Langfuse...")
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY")
    )
    print("✅ Langfuse initialized!")
    print()

    # Initialize agents
    print("🤖 Initializing specialized agents (takes ~15 seconds)...")
    print()

    print("   → HR Agent...")
    hr = HRAgent(langfuse_handler=langfuse_handler)
    hr.initialize('data/hr_docs')

    print("   → Tech/IT Agent...")
    tech = TechAgent(langfuse_handler=langfuse_handler)
    tech.initialize('data/tech_docs')

    print("   → Finance Agent...")
    finance = FinanceAgent(langfuse_handler=langfuse_handler)
    finance.initialize('data/finance_docs')

    print()
    print("✅ All agents initialized!")

    # Create orchestrator
    print("🎯 Creating orchestrator...")
    orchestrator = OrchestratorAgent(hr, tech, finance, langfuse_handler=langfuse_handler)

    # Create evaluator
    print("⭐ Creating evaluator (BONUS)...")
    evaluator = EvaluatorAgent(langfuse_client=langfuse)

    print()
    print("="*80)
    print("✅ SYSTEM READY!")
    print("="*80)
    print()

    return orchestrator, evaluator, langfuse

def process_query(orchestrator, evaluator, query):
    """Process a query and show results"""
    print()
    print("="*80)
    print(f"📝 QUERY: {query}")
    print("="*80)
    print()

    # Process query
    result = orchestrator.process_query(query, verbose=True)

    # Show results
    print()
    print("-"*80)
    print("📊 CLASSIFICATION RESULTS")
    print("-"*80)
    print(f"Department: {result['classification']['department']}")
    print(f"Confidence: {result['classification']['confidence']:.1%}")
    print(f"Reasoning: {result['classification']['reasoning']}")

    print()
    print("-"*80)
    print("💬 ANSWER")
    print("-"*80)
    print(result['answer'])

    # Evaluate quality
    print()
    print("-"*80)
    print("⭐ QUALITY EVALUATION (BONUS)")
    print("-"*80)

    evaluation = evaluator.evaluate_response(
        query=result["query"],
        answer=result["answer"],
        department=result["classification"]["department"],
        source_documents=result["source_documents"]
    )

    print(f"Overall Score:    {evaluation.overall_score}/10")
    print(f"Relevance:        {evaluation.relevance_score}/10")
    print(f"Completeness:     {evaluation.completeness_score}/10")
    print(f"Accuracy:         {evaluation.accuracy_score}/10")
    print(f"Clarity:          {evaluation.clarity_score}/10")
    print()
    print(f"Feedback: {evaluation.feedback}")

    print()
    print("="*80)
    print()

    return result, evaluation

def main():
    """Main interactive loop"""
    # Initialize system
    orchestrator, evaluator, langfuse = initialize_system()

    print("🔗 View traces at: https://cloud.langfuse.com")
    print()
    print("💡 Example questions:")
    print("   HR:      'How many PTO days do I get per year?'")
    print("   IT:      'My laptop won't turn on, what should I do?'")
    print("   Finance: 'What is the expense reimbursement policy?'")
    print()
    print("="*80)
    print()

    # Interactive loop
    while True:
        try:
            # Get user input
            query = input("❓ Enter your question (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print()
                print("👋 Thanks for using the system!")
                print("🔗 Don't forget to check Langfuse for traces: https://cloud.langfuse.com")
                print()
                break

            if not query:
                print("⚠️  Please enter a question!")
                continue

            # Process the query
            process_query(orchestrator, evaluator, query)

        except KeyboardInterrupt:
            print()
            print()
            print("👋 Interrupted. Exiting...")
            print()
            break
        except Exception as e:
            print()
            print(f"❌ Error: {e}")
            print("Please try again.")
            print()

if __name__ == "__main__":
    main()
