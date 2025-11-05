"""
Configuration module for the Travel Assistant.
Handles environment variables and LLM initialization.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0  # Deterministic outputs for better consistency
MAX_TOKENS = 2048

# File Paths
DATA_DIR = "data"
FLIGHTS_FILE = os.path.join(DATA_DIR, "flights.json")
VISA_RULES_FILE = os.path.join(DATA_DIR, "visa_rules.md")

# Vector Store Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store"


def get_llm():
    """
    Initialize and return the LLM instance.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your .env file.\n"
            "Get your free API key at: https://console.groq.com/"
        )
    
    return ChatGroq(
        model_name=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )


# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful travel assistant specializing in international flight bookings and travel policies.

Your capabilities:
1. Search for flights based on user preferences (destination, dates, airlines, layovers, budget)
2. Answer questions about visa requirements for different countries
3. Provide information about refund and cancellation policies
4. Explain airline alliance benefits

Guidelines:
- Always be friendly, professional, and concise
- When searching flights, extract all relevant criteria from the user's request
- If information is missing, ask clarifying questions
- Present flight results in a clear, organized format
- When answering policy questions, provide accurate information from the knowledge base
- If you don't know something, say so honestly

Available tools:
- flight_search: Search for flights based on criteria
- policy_search: Answer visa and refund policy questions
"""