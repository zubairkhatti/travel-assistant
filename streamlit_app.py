"""
Streamlit UI for Travel Assistant
Provides a web-based interface for the conversational travel assistant.
"""

import streamlit as st
import time
import re
from src.agent import TravelAgent

# Page configuration
st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'agent' not in st.session_state:
        with st.spinner("Initializing Travel Assistant..."):
            st.session_state.agent = TravelAgent()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True


def display_sidebar():
    """Display sidebar with information and sample queries."""
    with st.sidebar:
        st.title("🌍 Travel Assistant")
        st.markdown("---")
        
        st.subheader("Features")
        st.markdown("""
        - ✈️ Flight search with multiple filters
        - 🛂 Visa requirements information
        - 📋 Refund & cancellation policies
        - 🤝 Airline alliance benefits
        """)
        
        st.markdown("---")
        
        st.subheader("📝 Sample Queries")
        
        sample_queries = [
            "Find flights to Tokyo in August with Star Alliance",
            "Do UAE passport holders need a visa for Japan?",
            "Show me direct flights to Paris under $700",
            "What is the refund policy?",
            "Find refundable flights to New York"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.sample_query = query
        
        st.markdown("---")
        
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.reset_memory()
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("ℹ️ About")
        st.markdown("""
        This AI assistant helps you plan international travel using:
        - **LangChain**: Agent orchestration
        - **RAG**: Knowledge retrieval
        - **Llama 3.3 70B Versatile**: Natural language understanding
        """)


def display_chat_messages():
    """Display chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handle user input from chat interface or sample query."""
    user_input = None

    if 'sample_query' in st.session_state:
        user_input = st.session_state.sample_query
        del st.session_state.sample_query

    if prompt := st.chat_input("Ask me about flights, visas, or travel policies..."):
        user_input = prompt

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(user_input)

            # --- Streaming cleanly ---
            placeholder = st.empty()
            text = ""

            # Split by sentence for smoother markdown-safe streaming
            chunks = re.split(r'(?<=[.!?]) +', response)

            for chunk in chunks:
                text += chunk + " "
                formatted_text = text.replace("\n", "  \n").replace("1.", "\n1.").replace("$", "\$")  # proper markdown line breaks
                placeholder.markdown(formatted_text)
                time.sleep(0.2)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    """Main Streamlit app."""
    # Initialize
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Main content
    st.title("✈️ Travel Assistant")
    st.markdown("Ask me anything about flights, visa requirements, or travel policies!")
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.info("""
        👋 **Welcome!** I'm your AI travel assistant.
        
        I can help you:
        - Find flights based on your preferences
        - Answer visa requirement questions
        - Explain refund and cancellation policies
        - Provide airline alliance information
        
        Try a sample query from the sidebar or type your own question below!
        """)
    
    # Display chat history
    display_chat_messages()
    
    # Handle new input
    handle_user_input()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("""
        Please ensure:
        1. GROQ_API_KEY is set in your .env file
        2. All dependencies are installed: `pip install -r requirements.txt`
        3. Data files exist in the data/ directory
        """)