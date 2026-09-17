import os 
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict , Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage , SystemMessage
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages


# let's load the environment variables from the .env file
load_dotenv()

# llm 
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
       google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
)


# we will structure the input and output of the graph using pydantic models
class MessageClssifier(BaseModel):
     message_type: Literal["emotional" ,"logical"] = Field(
          ...,
          description=(
                "Classify the message as emotional or logical."
                )
     )


class State(TypedDict):
     messages:Annotated[list , add_messages]
     message_type: str|None


# we will create a graph that classifies the message as emotional or logical

def classifiy_message(state:State):
     last_message = state["messages"][-1]
     classifier_llm = llm.with_structured_output(MessageClssifier)

     result = classifier_llm.invoke(
          [ SystemMessage( 
               content=""" 
               Classify the user's message as one of: 
               - emotional: The user is asking for emotional support, discussing feelings, relationships, personal problems, stress, sadness, anxiety, or similar topics.
                - logical: The user is asking for facts, information, technical help, explanations, calculations, or practical solutions. 
                Return only the classification. """ ),
                  HumanMessage(content=last_message.content) 
                                     ]
               )
     return {
          "message_type":result.message_type
     }

#  we will create a graph that route s the message to the appropriate agent based on the classification
def router(state:State):
     message_type = state.get("message_type")
     if message_type  == "emotional":
          return "therapist"
     return "logical"

# Therapist Node

def therapist_agent(state:State):
     messages =[
           SystemMessage( 
                content=
                """ You are a compassionate and supportive assistant.
                  Focus on the emotional aspects of the user's message. 
                  Your response should: - Show empathy. 
                  - Acknowledge the user's feelings. 
                  - Help the user understand and process what they are feeling.
                    - Ask thoughtful questions when appropriate. Do not immediately jump into technical or purely logical solutions unless the user explicitly asks for them. """
                     )
            ]
     # messages history 
     messages.extend(state["messages"])
     reply = llm.invoke(messages)
     return {
          "messages":[reply]
     }

# logical Node
def logical_agent(state:State):
     messages =[ 
          SystemMessage(
                content=
                """ You are a logical and practical assistant. 
                Focus on: - Facts - Clear explanations - Reasoning - Evidence - Practical solutions Be direct and concise. 
                Do not focus on emotional support unless the user explicitly asks for it. """ 
                ) ]
     # messages history 
     messages.extend(state["messages"])
     reply = llm.invoke(messages)
     return {
          "messages":["this is logical" + reply]
     }


#  this will build our graph and return the graph object

grpah_build = StateGraph(State)


grpah_build.add_node("classifier" , classifiy_message)
grpah_build.add_node("therapist" , therapist_agent)
grpah_build.add_node("logical" , logical_agent)

grpah_build.add_edge(START , "classifier")
# grpah_build.add_edge("classifier" , "router") 
grpah_build.add_conditional_edges(
     "classifier" , 
     router,
     {
          "therapist":"therapist" , 
          "logical":"logical"
     }
)

grpah_build.add_edge("therapist" , END)
grpah_build.add_edge("logical" , END)



graph = grpah_build.compile()


# we will now invoke the graph with user input and print the response

def run_chatbot():
     state:State={
          "messages":[],
          "message_type":None

     }

     while True:
          user_input = input("message :")

          if user_input.lower() in ["exit" , "quit"]:
               print("Exiting the chatbot. Goodbye!")
               break

          state["messages"].append(
               HumanMessage(content=user_input)
          )

          state = graph.invoke(state)
          last_message = state["messages"][-1]
          print(f"Agent: {last_message.content}")


run_chatbot()