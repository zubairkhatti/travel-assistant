"""
Travel Assistant - Main Application
A conversational AI assistant for international travel planning.

This application demonstrates:
- Natural language query interpretation
- LangChain agent orchestration with ReAct prompting
- RAG-based policy question answering
- Flight search with multiple filter criteria
"""

import sys
from src.agent import TravelAgent


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 60)
    print("🌍 Welcome to the Travel Assistant! ✈️")
    print("=" * 60)
    print("\nI can help you with:")
    print("  • Finding flights based on your preferences")
    print("  • Answering visa requirement questions")
    print("  • Explaining refund and cancellation policies")
    print("  • Providing airline alliance information")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to clear conversation history.")
    print("=" * 60 + "\n")


def print_sample_queries():
    """Print sample queries for testing."""
    print("\n📝 Sample Queries to Try:")
    print("-" * 60)
    print("1. Find me a round-trip to Tokyo in August with Star Alliance")
    print("   airlines only. I want to avoid overnight layovers.")
    print("\n2. Do UAE passport holders need a visa for Japan?")
    print("\n3. Show me direct flights to Paris under $700")
    print("\n4. What is the refund policy for tickets?")
    print("\n5. Find flights to New York in August, refundable tickets only")
    print("-" * 60 + "\n")


def run_interactive_mode():
    """Run the assistant in interactive mode."""
    print_welcome()
    print("Initializing Travel Assistant...")
    
    try:
        agent = TravelAgent()
        print("✓ Assistant ready!\n")
        print_sample_queries()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Travel Assistant. Safe travels!")
                    break
                
                if user_input.lower() == 'reset':
                    agent.reset_memory()
                    continue
                
                if user_input.lower() == 'help':
                    print_sample_queries()
                    continue
                
                if not user_input:
                    continue
                
                # Get agent response
                print("\nAssistant: ", end="", flush=True)
                response = agent.chat(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}\n")
                continue
    
    except Exception as e:
        print(f"\n❌ Failed to initialize: {str(e)}")
        print("\nPlease ensure:")
        print("  1. GROQ_API_KEY is set in your .env file")
        print("  2. All required packages are installed (pip install -r requirements.txt)")
        print("  3. Data files exist in the data/ directory")
        sys.exit(1)


def run_demo_queries():
    """Run a set of demo queries to showcase capabilities."""
    print("\n" + "=" * 60)
    print("🎬 DEMO MODE - Running Sample Queries")
    print("=" * 60 + "\n")
    
    demo_queries = [
        "Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers.",
        "Do UAE passport holders need a visa for Japan?",
        "What is the refund policy for tickets?",
        "Show me direct flights to Paris",
    ]
    
    try:
        agent = TravelAgent()
        print("✓ Assistant initialized\n")
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*60}")
            print(f"Demo Query {i}/{len(demo_queries)}")
            print(f"{'='*60}")
            print(f"User: {query}\n")
            
            response = agent.chat(query)
            print(f"Assistant: {response}\n")
            
            input("Press Enter to continue to next query...")
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        run_demo_queries()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()