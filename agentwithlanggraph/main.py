from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph , START ,END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os 


load_dotenv()

# llm is brain of the agent. It is responsible for generating responses based on the input messages. The model used here is "gemini-3.6-flash", which is a Google Generative AI model.
chat = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
       google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
)


# State is a TypedDict that defines the structure of the state used in the graph.
#  It contains a single key "messages", which is a list of messages.
#  The add_messages function is used to annotate the type of the messages list, ensuring that it adheres to the expected format for messages in the graph.
# what is annotated? Annotated is a way to add metadata to types in Python. In this case, it is used to specify that the list of messages should be processed by the add_messages function, which likely adds additional validation or processing to the messages.
class State(TypedDict):
    messages: Annotated[list , add_messages]


# graph_builder is an instance of StateGraph, which is used to build a state machine or graph that defines the flow of the agent's interactions. 
# It takes the State TypedDict as its type parameter, ensuring that the states in the graph adhere to the defined structure.
graph_builder = StateGraph(State)

# chatbot is a function that takes the current state of the graph as input and generates a response using the chat model.
def chatbot(state:State):
    response = chat.invoke(state["messages"])
    return {"messages": [response]}

# The graph is constructed by adding nodes and edges.
#  The "chatbot" node is added to the graph, and edges are created from the START node to the "chatbot" node and from the "chatbot" node to the END node.
#  This defines the flow of the interaction, where the user input is processed by the chatbot, and the response is returned as the final output of the graph.
graph_builder.add_node("chatbot" , chatbot)
graph_builder.add_edge(START , "chatbot")
graph_builder.add_edge("chatbot" , END)

# this step is the final step where the graph is compiled, making it ready for execution.
#  The compiled graph can then be invoked with user input to generate responses based on the defined flow and the chat model's capabilities.
graph = graph_builder.compile()

user_input = input("User: ")

state = graph.invoke({
    "messages":
    [
        {
               "role": "user",
               "content": user_input
        }
    ]
})


print(state["messages"][-1].content)

