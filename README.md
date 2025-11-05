# 🌍 Travel Assistant - Conversational AI for International Travel Planning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated conversational AI assistant that helps users plan international travel through natural language interactions. Built with LangChain, RAG (Retrieval Augmented Generation), and advanced prompt engineering.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Sample Outputs](#sample-outputs)

## 🎯 Overview

This project demonstrates a production-ready conversational AI system that:
- **Interprets natural language queries** about travel preferences
- **Searches and filters flights** using multiple criteria (destination, dates, airlines, layovers, price)
- **Answers policy questions** using RAG with vector similarity search
- **Orchestrates multiple tools** through LangChain agent with ReAct prompting
- **Maintains conversation context** for multi-turn interactions

## ✨ Features

### Core Capabilities

1. **Natural Language Flight Search**
   - Parse complex queries: *"Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers."*
   - Extract and normalize search criteria (origin, destination, dates, airlines, alliances, layovers, price)
   - Apply intelligent filtering and ranking

2. **RAG-Powered Policy Q&A**
   - Answer visa requirements for different countries
   - Explain refund and cancellation policies
   - Provide airline alliance benefits
   - Vector-based semantic search with FAISS

3. **Agent-Based Orchestration**
   - LangChain ReAct agent for tool selection
   - Context-aware responses
   - Fallback handling for unclear queries
   - Conversation memory for follow-up questions

4. **User-Friendly Interfaces**
   - Command-line interactive mode
   - Streamlit web interface (optional)
   - Demo mode for testing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangChain Agent                           │
│                  (ReAct Prompting)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Query interpretation                                │   │
│  │ • Tool selection                                      │   │
│  │ • Context management                                  │   │
│  │ • Response generation                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌──────────────────────┐  ┌────────────────────────┐
│   Flight Search      │  │   Policy Search        │
│       Tool           │  │       Tool             │
│                      │  │                        │
│  • Parse criteria    │  │  • Vector search       │
│  • Filter flights    │  │  • RAG retrieval       │
│  • Rank results      │  │  • Answer generation   │
└──────────────────────┘  └────────────────────────┘
          │                         │
          ▼                         ▼
┌──────────────────────┐  ┌────────────────────────┐
│  flights.json        │  │  FAISS Vector Store    │
│  (Mock Flight Data)  │  │  (visa_rules.md)       │
└──────────────────────┘  └────────────────────────┘
```

### Component Breakdown

1. **LLM Layer**: Llama 3.3 70B Versatile via Groq API
2. **Agent Layer**: LangChain ReAct agent with memory
3. **Tool Layer**: Flight search + RAG policy search
4. **Data Layer**: JSON flight listings + Markdown knowledge base
5. **Embedding Layer**: Sentence Transformers for vector search

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key (free, no credit card required)

### Step 1: Clone the Repository

```bash
git clone https://github.com/zubairkhatti/travel-assistant.git
cd travel-assistant
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Get Groq API Key

1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account (no credit card required)
3. Create an API key

### Step 5: Configure Environment

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Step 6: Initialize Vector Store

The vector store will be created automatically on first run.

## 💻 Usage

### Interactive Mode (Recommended)

```bash
python main.py
```

This starts an interactive chat session where you can:
- Ask questions naturally
- Get flight recommendations
- Learn about visa requirements
- Clear conversation history with `reset`
- Exit with `quit` or `exit`

### Demo Mode

```bash
python main.py --demo
```

Runs predefined queries to showcase all capabilities.

### Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

Opens a web browser with an interactive UI featuring:
- Chat interface
- Sample query buttons
- Conversation history
- Clear/reset functionality

## 📁 Project Structure

```
travel-assistant/
├── main.py                     # Main CLI application
├── streamlit_app.py           # Web UI (optional)
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── .env.example              # Example environment file
├── .gitignore                # Git ignore rules
├── README.md                 # This file
│
├── data/                     # Data files
│   ├── flights.json          # Mock flight listings (20 flights)
│   └── visa_rules.md         # Travel policy knowledge base
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration and LLM setup
│   ├── utils.py              # Utility functions
│   ├── flight_search.py      # Flight search implementation
│   ├── rag_system.py         # RAG system with FAISS
│   └── agent.py              # LangChain agent orchestration
│
└── vector_store/             # FAISS index (auto-generated)
    ├── index.faiss
    └── index.pkl

```

## 🔧 Technical Implementation

### 1. Prompt Engineering Strategy

#### System Prompt Design

```python
SYSTEM_PROMPT = """You are a helpful travel assistant specializing in 
international flight bookings and travel policies.

Your capabilities:
1. Search for flights based on user preferences
2. Answer questions about visa requirements
3. Provide refund and cancellation policy information
4. Explain airline alliance benefits

Guidelines:
- Be friendly, professional, and concise
- Extract all relevant criteria before searching
- Ask clarifying questions when needed
- Present results in organized format
- Be honest if you don't know something
"""
```

**Key Techniques:**
- **Role definition**: Clear identity as travel assistant
- **Capability listing**: Explicit tool descriptions
- **Behavioral guidelines**: Tone and interaction rules
- **Fallback instructions**: Handle unknown queries gracefully

#### ReAct Prompting Pattern

The agent uses ReAct (Reasoning + Acting) prompting:

```
Thought: Do I need to use a tool? Yes
Action: flight_search
Action Input: "Tokyo in August with Star Alliance, no overnight layovers"
Observation: [Flight results]
Thought: I have the information needed
Final Answer: [Formatted response]
```

**Benefits:**
- Explicit reasoning steps
- Tool selection transparency
- Error recovery capability
- Audit trail for debugging

### 2. LangChain Agent Architecture

#### Agent Components

1. **Tools**
   - `flight_search`: Filters flight database
   - `policy_search`: RAG-based Q&A

2. **Memory**
   - `ConversationBufferMemory`: Maintains chat history
   - Enables follow-up questions
   - Context-aware responses

3. **Executor**
   - Manages tool invocation
   - Handles parsing errors
   - Limits iteration depth (max 5)

#### Tool Design Pattern

```python
Tool(
    name="flight_search",
    func=search_flights,
    description="""Detailed description of when and how to use this tool.
    
    Input format:
    - Expected parameters
    - Example queries
    
    Output format:
    - What the tool returns
    """
)
```

**Design Principles:**
- **Clear descriptions**: Help LLM choose correct tool
- **Input examples**: Guide query formation
- **Output specifications**: Set expectations

### 3. RAG System Implementation

#### Pipeline

1. **Document Loading**
   ```python
   # Load markdown file
   content = load_text_file("visa_rules.md")
   ```

2. **Text Splitting**
   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50
   )
   chunks = splitter.split_text(content)
   ```

3. **Embedding Generation**
   ```python
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   ```

4. **Vector Store Creation**
   ```python
   vector_store = FAISS.from_documents(documents, embeddings)
   ```

5. **Similarity Search**
   ```python
   relevant_docs = vector_store.similarity_search(query, k=3)
   ```

6. **Answer Generation**
   ```python
   prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
   response = llm.invoke(prompt)
   ```

#### Optimization Strategies

- **Chunk size**: 500 tokens (balance between context and specificity)
- **Overlap**: 50 tokens (preserve context across boundaries)
- **Top-k retrieval**: 3 documents (relevant without overwhelming)
- **Normalization**: Embeddings normalized for better similarity

### 4. Flight Search Logic

#### Multi-Criteria Filtering

```python
def search(
    origin, destination, departure_month, alliance,
    max_price, refundable_only, avoid_overnight_layover, max_layovers
):
    results = flights
    
    # Apply filters sequentially
    if origin: results = filter_by_origin(results, origin)
    if destination: results = filter_by_destination(results, destination)
    # ... more filters
    
    # Sort by price
    return sorted(results, key=lambda x: x['price_usd'])
```

#### Natural Language Parsing

```python
# Extract criteria from text
if 'star alliance' in query.lower():
    alliance = 'Star Alliance'

if 'avoid overnight' in query.lower():
    avoid_overnight = True

month, year = extract_month_year(query)
```

**Parsing Techniques:**
- Keyword matching for alliances
- Regex for price extraction
- Date parsing for flexible formats
- Negation handling ("avoid", "no")

### 5. LLM Provider Choice

**Why Llama 3.3 70B Versatile via Groq?**

1. **Quality**: 70B parameter model with GPT-4 class performance
2. **Accessibility**: Free tier, no credit card required
3. **Speed**: Groq's inference is extremely fast
4. **Reproducibility**: Anyone can run the project immediately
5. **Cost**: Zero cost for development and demo


### 6. Conversation Design & UX

#### Tone & Style
- **Friendly but professional**: "I'd be happy to help you find flights!"
- **Concise**: Avoid overwhelming with information
- **Structured**: Use formatting for readability

#### Fallback Handling

```python
try:
    response = agent.invoke(query)
except Exception as e:
    return "I apologize, I encountered an error. Please rephrase."
```

#### Clarification Strategy

When information is missing:
```
"I'd be happy to search for flights! To help you better, could you tell me:
- Your preferred travel dates?
- Any airline preferences?"
```

## 📊 Sample Outputs

### Example 1: Flight Search

**User Query:**
```
Find me a round-trip to Tokyo in August with Star Alliance airlines only. 
I want to avoid overnight layovers.
```

**Agent Response:**
```
I've found 5 round-trip flights to Tokyo in August with Star Alliance airlines. 
However, I noticed that Flight FL004 has an overnight layover, which you wanted to avoid.
Here are the remaining options:

1. Lufthansa (FL003) with layovers in Frankfurt and Munich, priced at $875 USD, but it's not refundable.
2. Asiana Airlines (FL020) with a layover in Seoul, priced at $910 USD, but it's not refundable.
3. Turkish Airlines (FL001) with a layover in Istanbul, priced at $950 USD, and it's refundable.
4. ANA (FL006) with a layover in Bangkok, priced at $1050 USD, and it's refundable.

Please let me know if you'd like me to refine the search or if you have any other questions!
```

### Example 2: Visa Question

**User Query:**
```
Do UAE passport holders need a visa for Japan?
```

**Agent Response:**
```
No, UAE passport holders do not need a visa to enter Japan for tourism purposes, 
as long as their stay is up to 30 days and their passport is valid for at least 6 months
beyond their intended stay.
```

### Example 3: Policy Question

**User Query:**
```
What's the refund policy for tickets?
```

**Agent Response:**
```
The refund policy for tickets depends on the type of ticket. For non-refundable tickets, 
they cannot be canceled for a cash refund, but changes to flight dates or requesting travel 
credit for future use may be possible. For refundable tickets, they can be canceled up to 
48 hours before departure with a 10% processing fee. Same-day cancellations within 24 hours 
of booking may be eligible for a full refund, depending on the airline's policy.
```
## 🔍 Troubleshooting

### Common Issues

**1. API Key Error**
```
ValueError: GROQ_API_KEY not found
```
**Solution**: Create `.env` file with your Groq API key

**2. Vector Store Error**
```
Error loading vector store
```
**Solution**: Delete `vector_store/` folder and restart (will rebuild automatically)

**3. Module Import Error**
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root directory, not from inside `src/`

**4. Slow First Run**
```
Downloading embeddings model...
```
**Solution**: Normal behavior - sentence-transformers downloads on first use (~100MB)

## 📝 License

MIT License - feel free to use for learning and development.

## 🙏 Acknowledgments

- **LangChain**: Agent orchestration framework
- **Groq**: Fast, free LLM inference
- **HuggingFace**: Embedding models
- **FAISS**: Efficient vector search

---

**Built with ❤️ for the KAVAK Technical Case Study**

*Demonstrating production-ready AI engineering with LangChain, RAG, and conversational AI best practices.*