"""
Orchestrator Agent - Routes user queries to appropriate specialized agents
Classifies user intent and delegates to HR, IT, or Finance agents.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langfuse.langchain import CallbackHandler
import os


class IntentClassification(BaseModel):
    """
    Structured output for intent classification.
    """
    department: Literal["HR", "IT", "Finance"] = Field(
        description="The department that should handle this query"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this department was chosen"
    )


class OrchestratorAgent:
    """
    Orchestrator Agent that classifies user intent and routes to specialized agents.
    """

    def __init__(self, hr_agent, tech_agent, finance_agent, langfuse_handler=None):
        """
        Initialize the Orchestrator with specialized agents.

        Args:
            hr_agent: Initialized HR agent instance
            tech_agent: Initialized Tech/IT agent instance
            finance_agent: Initialized Finance agent instance
            langfuse_handler: Optional Langfuse callback handler for tracing
        """
        self.hr_agent = hr_agent
        self.tech_agent = tech_agent
        self.finance_agent = finance_agent
        self.langfuse_handler = langfuse_handler

        # Get model name from environment variable
        llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

        # Add "openai/" prefix if not already present (for OpenRouter compatibility)
        if not llm_model.startswith("openai/") and not "/" in llm_model:
            llm_model = f"openai/{llm_model}"

        # Initialize the LLM for intent classification using OpenRouter
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,  # Zero temperature for consistent classification
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/assignment-ai",
                "X-Title": "Multi-Agent Support System"
            }
        )

        # Set up the output parser
        self.parser = PydanticOutputParser(pydantic_object=IntentClassification)

        # Create the classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent query router for a company support system.

Your job is to classify incoming user queries and route them to the appropriate department:

- HR: Questions about employee benefits, PTO, leave policies, onboarding, performance reviews, remote work policies, professional development, company policies
- IT: Questions about technical issues, software problems, hardware troubleshooting, network connectivity, passwords, VPN, email, security, device setup
- Finance: Questions about expenses, reimbursements, invoices, payments, budgets, procurement, vendors, financial policies

Analyze the user's query carefully and determine which department is best suited to handle it.

{format_instructions}"""),
            ("human", "{query}")
        ])

    def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify the user's query to determine the appropriate department.

        Args:
            query: The user's question

        Returns:
            IntentClassification with department, confidence, and reasoning
        """
        print(f"\n[Orchestrator] Classifying query: {query}")

        # Format the prompt with the parser instructions
        formatted_prompt = self.classification_prompt.format_messages(
            query=query,
            format_instructions=self.parser.get_format_instructions()
        )

        # Get classification from LLM
        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        response = self.llm.invoke(
            formatted_prompt,
            config={"callbacks": callbacks}
        )

        # Parse the response
        classification = self.parser.parse(response.content)

        print(f"[Orchestrator] Classified as: {classification.department} "
              f"(confidence: {classification.confidence:.2f})")
        print(f"[Orchestrator] Reasoning: {classification.reasoning}")

        return classification

    def route_query(self, query: str) -> dict:
        """
        Route the query to the appropriate specialized agent.

        Args:
            query: The user's question

        Returns:
            Dictionary with classification, answer, and metadata
        """
        # Classify the intent
        classification = self.classify_intent(query)

        # Route to the appropriate agent
        if classification.department == "HR":
            result = self.hr_agent.answer_query(query)
        elif classification.department == "IT":
            result = self.tech_agent.answer_query(query)
        elif classification.department == "Finance":
            result = self.finance_agent.answer_query(query)
        else:
            # Fallback (should not happen with Literal type)
            raise ValueError(f"Unknown department: {classification.department}")

        # Combine classification and agent response
        return {
            "query": query,
            "classification": {
                "department": classification.department,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning
            },
            "answer": result["answer"],
            "source_documents": result["source_documents"],
            "agent_used": result["agent"]
        }

    def process_query(self, query: str, verbose: bool = True) -> dict:
        """
        Process a user query end-to-end with optional verbose output.

        Args:
            query: The user's question
            verbose: Whether to print detailed processing information

        Returns:
            Dictionary with full query processing results
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"PROCESSING QUERY: {query}")
            print("=" * 80)

        result = self.route_query(query)

        # Capture trace ID if langfuse_handler is available
        if self.langfuse_handler and hasattr(self.langfuse_handler, 'get_trace_id'):
            result['trace_id'] = self.langfuse_handler.get_trace_id()

        if verbose:
            print("\n" + "-" * 80)
            print(f"CLASSIFICATION:")
            print(f"  Department: {result['classification']['department']}")
            print(f"  Confidence: {result['classification']['confidence']:.2f}")
            print(f"  Reasoning: {result['classification']['reasoning']}")
            print("-" * 80)
            print(f"\nANSWER:")
            print(result['answer'])
            print("\n" + "=" * 80 + "\n")

        return result
