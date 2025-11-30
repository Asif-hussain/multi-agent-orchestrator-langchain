"""
HR Agent - Specialized RAG agent for Human Resources queries
Handles questions about employee benefits, PTO, policies, and HR procedures.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler
import os


class HRAgent:
    """
    HR Agent specializing in employee-related queries.
    Uses RAG to provide accurate answers grounded in company HR documentation.
    """

    def __init__(self, langfuse_handler=None):
        """
        Initialize the HR Agent with document retrieval and LLM.

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
            temperature=0.1,  # Low temperature for factual responses
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={
                "HTTP-Referer": "https://github.com/yourusername/assignment-ai",
                "X-Title": "Multi-Agent Support System"
            }
        )
        self.vector_store = None
        self.qa_chain = None

    def load_documents(self, docs_path="data/hr_docs"):
        """
        Load HR documentation and create vector store.

        Args:
            docs_path: Path to HR documentation directory
        """
        print(f"Loading HR documents from {docs_path}...")

        # Load all text files from the HR docs directory
        loader = DirectoryLoader(
            docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()

        print(f"Loaded {len(documents)} HR documents")

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
            collection_name="hr_docs"
        )

        print("HR vector store created successfully")

    def create_qa_chain(self):
        """
        Create the RetrievalQA chain with custom prompt.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_documents first.")

        # Custom prompt template for HR queries
        template = """You are an HR specialist assistant for the company. Use the following pieces of context from the company's HR documentation to answer the question at the end.

If you don't find the exact answer in the context, say so honestly and suggest who the user should contact (HR department).

Always be professional, empathetic, and accurate. For policy questions, cite the specific policy when possible.

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

        print("HR QA chain created successfully")

    def answer_query(self, query: str) -> dict:
        """
        Answer an HR-related query using RAG.

        Args:
            query: The user's HR question

        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")

        print(f"\n[HR Agent] Processing query: {query}")

        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        result = self.qa_chain.invoke(
            {"query": query},
            config={"callbacks": callbacks}
        )

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"],
            "agent": "HR"
        }

    def initialize(self, docs_path="data/hr_docs"):
        """
        Convenience method to initialize the entire agent.

        Args:
            docs_path: Path to HR documentation directory
        """
        self.load_documents(docs_path)
        self.create_qa_chain()
        print("HR Agent initialized and ready!")
