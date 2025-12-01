"""
Finance Agent - Specialized RAG agent for Finance queries
Handles questions about expenses, budgets, invoices, and financial procedures.
"""

import logging
import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler

# Import config from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)

# Prompt template for Finance queries
FINANCE_PROMPT_TEMPLATE = """You are a Finance department specialist for the company. Use the following pieces of context from the company's financial documentation to answer the question at the end.

Provide accurate information about expenses, budgets, invoices, and financial policies. Include specific amounts, deadlines, and approval requirements when mentioned in the documentation.

If you don't find the exact answer in the context, say so honestly and suggest contacting the Finance department.

Be precise with numbers, dates, and policy details.

Context:
{context}

Question: {question}

Helpful Answer:"""


class FinanceAgent:
    """
    Finance Agent specializing in financial queries.
    Uses RAG to provide accurate answers grounded in company financial documentation.
    """

    def __init__(self, langfuse_handler: Optional[CallbackHandler] = None):
        """
        Initialize the Finance Agent with document retrieval and LLM.

        Args:
            langfuse_handler: Optional Langfuse callback handler for tracing
        """
        self.langfuse_handler = langfuse_handler
        self.vector_store = None
        self.qa_chain = None

        self.embeddings = self._setup_embeddings()
        self.llm = self._setup_llm()

        logger.info("Finance Agent instance created")

    def _normalize_model_name(self, model: str) -> str:
        """Ensure model name has correct prefix for OpenRouter compatibility."""
        if not model.startswith("openai/") and "/" not in model:
            return f"openai/{model}"
        return model

    def _setup_embeddings(self) -> OpenAIEmbeddings:
        """Configure and return embeddings model."""
        embedding_model = Config.get_embedding_model()

        return OpenAIEmbeddings(
            openai_api_base=Config.get_openrouter_base_url(),
            openai_api_key=Config.get_openrouter_api_key(),
            model=embedding_model
        )

    def _setup_llm(self) -> ChatOpenAI:
        """Configure and return language model."""
        llm_model = Config.get_llm_model()
        llm_model = self._normalize_model_name(llm_model)

        return ChatOpenAI(
            model=llm_model,
            temperature=Config.get_temperature(),
            openai_api_base=Config.get_openrouter_base_url(),
            openai_api_key=Config.get_openrouter_api_key(),
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/assignment-ai",
                "X-Title": "Multi-Agent Support System"
            }
        )

    def load_documents(self, docs_path: str = "data/finance_docs") -> None:
        """
        Load Finance documentation and create vector store.

        Args:
            docs_path: Path to Finance documentation directory

        Raises:
            FileNotFoundError: If the documents path doesn't exist
            ValueError: If no documents are found
        """
        if not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"Finance documents path '{docs_path}' does not exist. "
                "Please ensure the data directory is properly set up."
            )

        logger.info(f"Loading Finance documents from {docs_path}")

        loader = DirectoryLoader(
            docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()

        if not documents:
            raise ValueError(
                f"No documents found in {docs_path}. "
                "Please add Finance documentation files."
            )

        logger.info(f"Loaded {len(documents)} Finance documents")

        chunks = self._split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="finance_docs"
        )

        logger.info("Finance vector store created successfully")

    def _split_documents(self, documents: list) -> list:
        """
        Split documents into chunks for optimal retrieval.

        Args:
            documents: List of loaded documents

        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.get_chunk_size(),
            chunk_overlap=Config.get_chunk_overlap(),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_qa_chain(self) -> None:
        """
        Create the RetrievalQA chain with custom prompt.

        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Call load_documents() before creating QA chain."
            )

        prompt = PromptTemplate(
            template=FINANCE_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": Config.get_retrieval_k()}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            callbacks=callbacks
        )

        logger.info("Finance QA chain created successfully")

    def answer_query(self, query: str) -> dict:
        """
        Answer a Finance-related query using RAG.

        Args:
            query: The user's Finance question

        Returns:
            Dictionary containing:
                - answer: The generated response
                - source_documents: List of relevant source documents
                - agent: Agent identifier

        Raises:
            ValueError: If QA chain is not initialized or query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self.qa_chain is None:
            raise ValueError(
                "QA chain not initialized. Call create_qa_chain() before answering queries."
            )

        logger.info(f"Processing Finance query: {query[:100]}...")

        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        result = self.qa_chain.invoke(
            {"query": query},
            config={"callbacks": callbacks}
        )

        logger.info("Query processed successfully")

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "agent": "Finance"
        }

    def initialize(self, docs_path: str = "data/finance_docs") -> None:
        """
        Convenience method to initialize the entire agent in one call.

        This method loads documents and creates the QA chain, preparing
        the agent to answer queries.

        Args:
            docs_path: Path to Finance documentation directory

        Raises:
            FileNotFoundError: If docs_path doesn't exist
            ValueError: If no documents are found
        """
        logger.info("Initializing Finance Agent...")
        self.load_documents(docs_path)
        self.create_qa_chain()
        logger.info("Finance Agent initialized and ready!")

    @property
    def is_ready(self) -> bool:
        """Check if the agent is fully initialized and ready to answer queries."""
        return self.vector_store is not None and self.qa_chain is not None
