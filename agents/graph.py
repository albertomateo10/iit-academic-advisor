import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

# Import the State and Tools we just built
from state import AcademicAdvisorState
from tools import search_iit_courses_tool

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 1. Initialize the LLM 
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.1
)

# 2. Bind the tools to the LLM so it knows what "Hands" it has
tools = [search_iit_courses_tool]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the System Prompt (The "Proactive" Secret Sauce)
ADVISOR_SYSTEM_PROMPT = """You are the Proactive Academic Advisor for the Information Technology Management (ITM) department at the Illinois Institute of Technology (IIT).
Your goal is to help students navigate course catalogs and validate graduation requirements.

CRITICAL RULES:
1. DOMAIN GUARDRAIL: You are STRICTLY an academic advisor for IIT. If a student asks about topics unrelated to IIT, the ITM department, courses, or academic advising (e.g., general trivia, writing essays, recipes, coding help), you MUST politely refuse to answer. Steer the conversation back to academic advising.
2. ALWAYS use the `search_iit_courses_tool` to look up courses. DO NOT guess or hallucinate course codes, credits, or prerequisites.
3. PROACTIVE PREREQUISITE CHECKING: When a student asks to take a course, you MUST review the tool's output for prerequisites. 
4. If a prerequisite exists, check the student's 'completed_courses' list in your memory. If the prerequisite is NOT in their completed courses, DO NOT tell them they can enroll. Instead, pause and ask: "I see [Course] requires [Prerequisite]. Have you completed it?"
5. STRICT PERSONA: You are a human faculty advisor. NEVER use phrases like "Based on the search results," "According to the tool," or "I found." Speak directly and naturally.
   * BAD RESPONSE: "Based on the search results, the prerequisites are BIOL 445."
   * GOOD RESPONSE: "The prerequisites for BIOL 503 are BIOL 445."
"""

# --- DEFINE THE NODES ---

def call_model(state: AcademicAdvisorState):
    """This node invokes the Llama 3.1 model."""
    messages = state.get("messages", [])
    
    # We dynamically inject the student's completed courses into the system prompt
    completed_courses = state.get("completed_courses", [])
    context_prompt = ADVISOR_SYSTEM_PROMPT + f"\nStudent's Completed Courses: {completed_courses}"
    
    # Ensure the system prompt is always the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=context_prompt)] + messages
    else:
        # Update the existing system message with the latest completed courses
        messages[0] = SystemMessage(content=context_prompt)

    # Call the LLM
    response = llm_with_tools.invoke(messages)
    
    # Return the new message to be appended to the state
    return {"messages": [response]}

# We use LangGraph's prebuilt ToolNode to execute our Python search function
tool_node = ToolNode(tools)


# --- BUILD THE GRAPH ---

workflow = StateGraph(AcademicAdvisorState)

# Add our two main nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entry point
workflow.add_edge(START, "agent")

# Add the conditional logic:
# If the LLM output includes a tool call, route to the 'tools' node.
# If it's just regular text, route to END (meaning the agent replies to the user).
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# After the tool runs, always route back to the agent so it can read the DB results
workflow.add_edge("tools", "agent")

# Compile the final graph!
advisor_agent = workflow.compile()



if __name__ == "__main__":
    print("Welcome to the IIT Proactive Advisor CLI!")
    
    # Initialize an empty state for a new student
    initial_state = {
        "messages": [],
        "completed_courses": [],
        "current_gpa": 4.0,
        "awaiting_prereq_confirmation": False
    }
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        initial_state["messages"].append(HumanMessage(content=user_input))
        
        # Run the LangGraph state machine
        result = advisor_agent.invoke(initial_state)
        
        # Get the final response from the agent
        final_message = result["messages"][-1].content
        print(f"\nAdvisor: {final_message}")
        
        # Update our running state
        initial_state["messages"] = result["messages"]