"""
LangChain agent implementation for the Travel Assistant.
Orchestrates tools and manages conversation flow.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from src.config import get_llm, SYSTEM_PROMPT
from src.flight_search import create_flight_search_tool
from src.rag_system import create_policy_search_tool


class TravelAgent:
    """Main agent for handling travel queries."""
    
    def __init__(self):
        """Initialize the travel agent with LLM, tools, and memory."""
        self.llm = get_llm()
        self.tools = self._create_tools()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> list:
        """
        Create tools for the agent.
        
        Returns:
            List of Tool objects
        """
        # Flight search tool
        flight_search_func = create_flight_search_tool()
        
        # Policy search tool
        policy_search_func = create_policy_search_tool(self.llm)
        
        tools = [
            Tool(
                name="flight_search",
                func=flight_search_func,
                description="""Useful for searching flights based on user criteria.
Input should be a natural language query containing travel preferences such as:
- Origin and destination cities
- Travel dates or months (e.g., "August", "next month")
- Airline or alliance preferences (e.g., "Star Alliance")
- Price constraints (e.g., "under $1000")
- Layover preferences (e.g., "avoid overnight layovers", "direct flights")

Example inputs:
- "Find flights from Dubai to Tokyo in August with Star Alliance"
- "Show me direct flights to Paris under $700"
- "Round trip to New York avoiding overnight layovers"
"""
            ),
            Tool(
                name="policy_search",
                func=policy_search_func,
                description="""Useful for answering questions about travel policies, visa requirements, and regulations.
Use this tool when users ask about:
- Visa requirements for specific countries
- Passport validity rules
- Refund and cancellation policies
- Transit visa requirements
- Airline alliance benefits
- Travel insurance information
- COVID-19 or health requirements

Example inputs:
- "Do UAE passport holders need a visa for Japan?"
- "What is the refund policy for tickets?"
- "What are the benefits of Star Alliance?"
"""
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create the agent executor with ReAct prompting.
        
        Returns:
            AgentExecutor instance
        """
        # Create prompt template for ReAct agent
        template = f"""{SYSTEM_PROMPT}

TOOLS:
------
You have access to the following tools:

{{tools}}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{{chat_history}}

New input: {{input}}
{{agent_scratchpad}}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )
        
        return agent_executor
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and return agent response.
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's response
        """
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "I apologize, but I couldn't process your request.")
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."
    
    def reset_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        print("✓ Conversation history cleared")