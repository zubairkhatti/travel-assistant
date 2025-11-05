"""
RAG (Retrieval Augmented Generation) system for policy questions.
Uses FAISS vector store with sentence transformers for embeddings.
"""

import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.config import VISA_RULES_FILE, EMBEDDING_MODEL, VECTOR_STORE_PATH
from src.utils import load_text_file


class RAGSystem:
    """Handles document retrieval and question answering."""
    
    def __init__(self):
        """Initialize the RAG system with embeddings and vector store."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Load or create the vector store."""
        # Check if vector store exists
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✓ Loaded existing vector store")
                return
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
        
        # Create new vector store
        print("Creating new vector store...")
        documents = self._load_documents()
        
        if not documents:
            raise ValueError("No documents found to create vector store")
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save vector store
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        self.vector_store.save_local(VECTOR_STORE_PATH)
        print("✓ Created and saved new vector store")
    
    def _load_documents(self) -> List[Document]:
        """
        Load and split documents for the knowledge base.
        
        Returns:
            List of Document objects
        """
        # Load visa rules
        visa_content = load_text_file(VISA_RULES_FILE)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_text(visa_content)
        
        # Create Document objects
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": "visa_rules.md"}
            )
            for chunk in chunks
        ]
        
        return documents
    
    def search(self, query: str, k: int = 3) -> str:
        """
        Search for relevant information in the knowledge base.
        
        Args:
            query: User question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Retrieved context as a string
        """
        if not self.vector_store:
            return "Knowledge base not initialized."
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Combine retrieved content
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def answer_question(self, query: str, llm) -> str:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
            llm: Language model instance
            
        Returns:
            Answer to the question
        """
        # Retrieve relevant context
        context = self.search(query, k=3)
        
        # Create prompt with context
        prompt = f"""Based on the following information, answer the user's question.
If the information doesn't contain the answer, say so honestly.

Context:
{context}

Question: {query}

Answer:"""
        
        # Get response from LLM
        response = llm.invoke(prompt)
        return response.content


def create_policy_search_tool(llm):
    """
    Create a policy search tool for LangChain agent.
    
    Args:
        llm: Language model instance
        
    Returns:
        Function that searches policy information
    """
    rag_system = RAGSystem()
    
    def search_policies(query: str) -> str:
        """
        Search for visa requirements, refund policies, and other travel regulations.
        
        Use this tool when users ask about:
        - Visa requirements for specific countries
        - Passport validity requirements
        - Refund and cancellation policies
        - Transit visa requirements
        - Airline alliance benefits
        - Travel insurance information
        
        Args:
            query: User question about policies
            
        Returns:
            Answer based on the knowledge base
        """
        return rag_system.answer_question(query, llm)
    
    return search_policies