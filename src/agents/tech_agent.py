"""
Tech/IT Agent - Specialized RAG agent for IT Support queries
Handles questions about technical issues, software, hardware, and IT procedures.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler
import os


class TechAgent:
    """
    Tech/IT Agent specializing in technical support queries.
    Uses RAG to provide accurate answers grounded in IT support documentation.
    """

    def __init__(self, langfuse_handler=None):
        """
        Initialize the Tech Agent with document retrieval and LLM.

        Args:
            langfuse_handler: Optional Langfuse callback handler for tracing
        """
        self.langfuse_handler = langfuse_handler

        # Get model names from environment variables
        embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
        llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

        # Add "openai/" prefix if not already present (for OpenRouter compatibility)
        if not llm_model.startswith("openai/") and not "/" in llm_model:
            llm_model = f"openai/{llm_model}"

        # Use OpenRouter for embeddings (via OpenAI-compatible API)
        self.embeddings = OpenAIEmbeddings(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model=embedding_model
        )

        # Use OpenRouter for LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.1,  # Low temperature for factual, technical responses
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/assignment-ai",
                "X-Title": "Multi-Agent Support System"
            }
        )
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self, docs_path="data/tech_docs"):
        """
        Load IT documentation and create vector store.

        Args:
            docs_path: Path to IT documentation directory
        """
        print(f"Loading IT documents from {docs_path}...")

        # Load all text files from the tech docs directory
        loader = DirectoryLoader(
            docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()

        print(f"Loaded {len(documents)} IT documents")

        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Split into {len(chunks)} chunks")

        # Create vector store with embeddings
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="tech_docs"
        )

        print("IT vector store created successfully")

    def create_qa_chain(self):
        """
        Create the RetrievalQA chain with custom prompt.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_documents first.")

        # Custom prompt template for IT support queries
        template = """You are an IT support specialist for the company. Use the following pieces of context from the company's IT documentation to answer the question at the end.

Provide clear, step-by-step troubleshooting guidance when applicable. If you don't find the exact solution in the context, suggest general troubleshooting steps and recommend contacting IT Support.

Be technical but accessible. Include specific commands, settings, or procedures when mentioned in the documentation.

Context:
{context}

Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create retrieval QA chain
        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
            callbacks=callbacks
        )

        print("IT QA chain created successfully")

    def answer_query(self, query: str) -> dict:
        """
        Answer an IT-related query using RAG.

        Args:
            query: The user's IT question

        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")

        print(f"\n[Tech Agent] Processing query: {query}")

        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        result = self.qa_chain.invoke(
            {"query": query},
            config={"callbacks": callbacks}
        )

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "agent": "IT"
        }

    def initialize(self, docs_path="data/tech_docs"):
        """
        Convenience method to initialize the entire agent.

        Args:
            docs_path: Path to IT documentation directory
        """
        self.load_documents(docs_path)
        self.create_qa_chain()
        print("Tech Agent initialized and ready!")
