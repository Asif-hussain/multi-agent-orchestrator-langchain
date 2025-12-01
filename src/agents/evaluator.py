"""
Evaluator Agent - Assesses response quality and logs scores to Langfuse
Evaluates responses on multiple dimensions and provides quality scores.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from langfuse import Langfuse
import os


class QualityEvaluation(BaseModel):
    """
    Structured output for response quality evaluation.
    """
    overall_score: int = Field(
        description="Overall quality score from 1-10",
        ge=1,
        le=10
    )
    relevance_score: int = Field(
        description="How relevant is the answer to the question (1-10)",
        ge=1,
        le=10
    )
    completeness_score: int = Field(
        description="How complete and thorough is the answer (1-10)",
        ge=1,
        le=10
    )
    accuracy_score: int = Field(
        description="How accurate is the information (1-10)",
        ge=1,
        le=10
    )
    clarity_score: int = Field(
        description="How clear and understandable is the answer (1-10)",
        ge=1,
        le=10
    )
    feedback: str = Field(
        description="Detailed feedback explaining the scores"
    )
    strengths: str = Field(
        description="What the answer does well"
    )
    improvements: str = Field(
        description="What could be improved"
    )


class EvaluatorAgent:
    """
    Evaluator Agent that assesses response quality and logs to Langfuse.
    """

    def __init__(self, langfuse_client: Optional[Langfuse] = None):
        """
        Initialize the Evaluator Agent.

        Args:
            langfuse_client: Optional Langfuse client for logging scores
        """
        self.langfuse = langfuse_client

        # Get evaluator model name from environment variable
        evaluator_model = os.getenv("EVALUATOR_MODEL", "gpt-3.5-turbo")

        # Add "openai/" prefix if not already present (for OpenRouter compatibility)
        if not evaluator_model.startswith("openai/") and not "/" in evaluator_model:
            evaluator_model = f"openai/{evaluator_model}"

        # Initialize the LLM for evaluation using OpenRouter
        self.llm = ChatOpenAI(
            model=evaluator_model,
            temperature=0.2,  # Low temperature for consistent evaluation
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/assignment-ai",
                "X-Title": "Multi-Agent Support System"
            }
        )

        # Set up the output parser
        self.parser = PydanticOutputParser(pydantic_object=QualityEvaluation)

        # Create the evaluation prompt
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance evaluator for an AI support system.

Your job is to evaluate the quality of AI-generated responses to user queries. Assess each response on multiple dimensions:

1. **Relevance (1-10)**: Does the answer directly address the user's question?
2. **Completeness (1-10)**: Is the answer thorough and complete, or are important details missing?
3. **Accuracy (1-10)**: Is the information factually correct based on the context provided?
4. **Clarity (1-10)**: Is the answer clear, well-organized, and easy to understand?
5. **Overall Score (1-10)**: Your overall assessment of answer quality.

Scoring guidelines:
- 9-10: Excellent - Comprehensive, accurate, highly relevant
- 7-8: Good - Solid answer with minor gaps
- 5-6: Adequate - Answers the question but lacks detail or clarity
- 3-4: Poor - Significant issues with accuracy, relevance, or completeness
- 1-2: Very Poor - Does not adequately address the question

Provide detailed feedback explaining your scores, highlighting strengths, and suggesting improvements.

{format_instructions}"""),
            ("human", """Evaluate this response:

Original Question: {query}

Department Classified: {department}

AI Response: {answer}

Context Used:
{context}

Please provide your evaluation:""")
        ])

    def evaluate_response(
        self,
        query: str,
        answer: str,
        department: str,
        source_documents: list,
        trace_id: Optional[str] = None,
        observation_id: Optional[str] = None
    ) -> QualityEvaluation:
        """
        Evaluate the quality of a response.

        Args:
            query: The original user question
            answer: The AI-generated answer
            department: The department that handled the query
            source_documents: The source documents used for the answer
            trace_id: Optional Langfuse trace ID for linking evaluation

        Returns:
            QualityEvaluation with scores and feedback
        """
        print(f"\n[Evaluator] Evaluating response for query: {query[:50]}...")

        # Extract context from source documents
        context = "\n\n".join([
            f"Source {i+1}: {doc.page_content[:200]}..."
            for i, doc in enumerate(source_documents[:3])
        ])

        # Format the prompt
        formatted_prompt = self.evaluation_prompt.format_messages(
            query=query,
            answer=answer,
            department=department,
            context=context,
            format_instructions=self.parser.get_format_instructions()
        )

        # Get evaluation from LLM
        response = self.llm.invoke(formatted_prompt)

        # Parse the evaluation
        evaluation = self.parser.parse(response.content)

        print(f"[Evaluator] Overall Score: {evaluation.overall_score}/10")
        print(f"[Evaluator] Relevance: {evaluation.relevance_score}/10")
        print(f"[Evaluator] Completeness: {evaluation.completeness_score}/10")
        print(f"[Evaluator] Accuracy: {evaluation.accuracy_score}/10")
        print(f"[Evaluator] Clarity: {evaluation.clarity_score}/10")

        # Log to Langfuse if client is available
        if self.langfuse:
            if trace_id:
                self._log_to_langfuse(evaluation, trace_id)
            elif observation_id:
                self._log_to_langfuse_observation(evaluation, observation_id)
            else:
                # Create a standalone score event
                self._log_standalone_score(evaluation, query)

        return evaluation

    def _log_to_langfuse(self, evaluation: QualityEvaluation, trace_id: str):
        """
        Log evaluation scores to Langfuse.

        Args:
            evaluation: The quality evaluation
            trace_id: Langfuse trace ID to associate scores with
        """
        try:
            # Log overall score
            self.langfuse.create_score(
                trace_id=trace_id,
                name="overall_quality",
                value=evaluation.overall_score,
                comment=evaluation.feedback,
                data_type="NUMERIC"
            )

            # Log individual dimension scores
            self.langfuse.create_score(
                trace_id=trace_id,
                name="relevance",
                value=evaluation.relevance_score,
                comment="Relevance assessment",
                data_type="NUMERIC"
            )

            self.langfuse.create_score(
                trace_id=trace_id,
                name="completeness",
                value=evaluation.completeness_score,
                comment="Completeness assessment",
                data_type="NUMERIC"
            )

            self.langfuse.create_score(
                trace_id=trace_id,
                name="accuracy",
                value=evaluation.accuracy_score,
                comment="Accuracy assessment",
                data_type="NUMERIC"
            )

            self.langfuse.create_score(
                trace_id=trace_id,
                name="clarity",
                value=evaluation.clarity_score,
                comment="Clarity assessment",
                data_type="NUMERIC"
            )

            # Flush to ensure data is sent
            self.langfuse.flush()
            print(f"[Evaluator] Scores logged to Langfuse (trace_id: {trace_id})")

        except Exception as e:
            print(f"[Evaluator] Warning: Failed to log to Langfuse: {e}")
            import traceback
            traceback.print_exc()

    def _log_to_langfuse_observation(self, evaluation: QualityEvaluation, observation_id: str):
        """
        Log evaluation scores to Langfuse using observation ID.

        Args:
            evaluation: The quality evaluation
            observation_id: Langfuse observation ID to associate scores with
        """
        try:
            self.langfuse.create_score(
                observation_id=observation_id,
                name="overall_quality",
                value=evaluation.overall_score,
                comment=evaluation.feedback,
                data_type="NUMERIC"
            )
            print(f"[Evaluator] Scores logged to Langfuse (observation_id: {observation_id})")
        except Exception as e:
            print(f"[Evaluator] Warning: Failed to log to Langfuse: {e}")

    def _log_standalone_score(self, evaluation: QualityEvaluation, query: str):
        """
        Create standalone score event in Langfuse.

        Since we don't have a trace_id, scores won't be visible in the Scores tab.
        This is a limitation - scores must be associated with a trace to appear.

        Args:
            evaluation: The quality evaluation
            query: The query being evaluated
        """
        print(f"[Evaluator] Warning: Cannot log scores without trace_id.")
        print(f"[Evaluator] Scores will not appear in Langfuse dashboard.")
        print(f"[Evaluator] To fix: Pass trace_id when calling evaluate_response().")
        print(f"[Evaluator] Scores calculated: Overall={evaluation.overall_score}/10")

    def evaluate_and_log(
        self,
        result: dict,
        trace_id: Optional[str] = None
    ) -> QualityEvaluation:
        """
        Convenience method to evaluate a full query result.

        Args:
            result: The result dictionary from orchestrator
            trace_id: Optional Langfuse trace ID

        Returns:
            QualityEvaluation with scores and feedback
        """
        return self.evaluate_response(
            query=result["query"],
            answer=result["answer"],
            department=result["classification"]["department"],
            source_documents=result["source_documents"],
            trace_id=trace_id
        )
