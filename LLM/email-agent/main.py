"""
Main email assistant agent application.
Uses LangGraph for orchestrating email management tools.
"""

import logging
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from email_tools import EMAIL_TOOLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("email_assistant.log")],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_MODEL_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = init_chat_model(model=CHAT_MODEL, api_key=GEMINI_API_KEY)
llm_with_tools = llm.bind_tools(EMAIL_TOOLS)


# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Create chatbot node
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Create tool node
tool_node = ToolNode(EMAIL_TOOLS)

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()


def process_user_input(user_input: str):
    """Process user input and stream responses."""
    events = graph.stream({"messages": [{"role": "user", "content": user_input}]})
    for event in events:
        for value in event.values():
            if "messages" in value and value["messages"]:
                print("Assistant:", value["messages"][-1].content)


def main():
    """Main function to run the email assistant."""
    print("Email Assistant Agent started!")
    print("You can ask me to:")
    print("- List unread emails")
    print("- Get email content by UID")
    print("- Mark emails as read")
    print("- Summarize emails by UID")
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            process_user_input(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            print("Error:", str(e))
            break


if __name__ == "__main__":
    main()
