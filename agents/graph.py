import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Import the State and Tools we just built
from agents.state import AcademicAdvisorState
from agents.tools import search_iit_courses_tool, search_iit_policies_tool, search_iit_programs_tool

load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# # 1. Initialize the LLM 
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     # model="qwen/qwen3-32b",
#     # model="meta-llama/llama-4-scout-17b-16e-instruct",
#     api_key=GROQ_API_KEY,
#     temperature=0.1
# )
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# 1. Initialize the LLM with Claude
llm = ChatAnthropic(
    # model="claude-sonnet-4-20250514", 
    model="claude-haiku-4-5-20251001",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.1,
    max_tokens=1024
)

# 2. Bind the tools to the LLM so it knows what "Hands" it has
tools = [search_iit_courses_tool, search_iit_policies_tool, search_iit_programs_tool]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the System Prompt (Optimized for Haiku and natural conversation)
ADVISOR_SYSTEM_PROMPT = """You are an IIT ITM department academic advisor. Help students with courses, prerequisites, degree planning, and academic policies.

TOOL USAGE RULES:
- Use tools to look up courses, degree programs, or policies. Never guess course codes, credits, or prerequisites.
- Call each tool ONCE per question. If the result does not contain the answer, tell the student you do not have that information in your catalog. Do NOT call the same tool again with a rephrased query.
- If a question is about scheduling, calendars, tuition, or anything not in the course catalog or academic policies, say: "I don't have that information available. Please contact the ITM department or check the IIT website."

PREREQUISITES AND ENROLLMENT RULES (CRITICAL):
- When a student asks about a course's prerequisites, ALWAYS report them explicitly by name and code. If a course has no prerequisites, explicitly say "This course has no prerequisites."
- Read the raw prerequisite text carefully for AND/OR logic:
  * "AND" means the student must complete ALL listed courses.
  * "OR" means the student only needs ONE of the listed courses.
- When a student asks if they can take a course next semester, compare the course's prerequisites against the student's completed courses list.
- If they meet all requirements, enthusiastically confirm they can enroll.
- If ANY required prerequisite is NOT in their completed courses, do NOT confirm they can enroll. Instead explain: "You cannot take this course yet because it requires [prerequisite]. Have you completed it?"

DEGREE PLANNING AND RECOMMENDATIONS:
- When a student asks about a specialization or degree program, use your tools to find the curriculum and provide a clear, well-organized response that separates the "Core/Required Courses" from the "Elective Courses".
- When a student asks "Which course should I select next semester?", use the following logic to provide a thoughtful recommendation:
  1. Identify their specialization or degree.
  2. Find the required courses they have NOT taken yet (by comparing the degree requirements to their completed courses).
  3. Check the prerequisites for those missing courses.
  4. Recommend 1 or 2 courses that they are fully eligible to take right now, and elaborate slightly on why they are good options for their academic path.

OTHER RULES:
- Only answer IIT academic questions. Politely refuse unrelated topics.
- Speak naturally, warmly, and helpfully as a human faculty advisor. Provide well-elaborated answers. 
- Never say "based on the search results" or "according to the tool"."""

# --- DEFINE THE NODES ---

def call_model(state: AcademicAdvisorState):
    """This node invokes the LLM with token-managed message history."""
    messages = state.get("messages", [])

    # Dynamically inject the student's completed courses into the system prompt
    completed_courses = state.get("completed_courses", [])
    context_prompt = ADVISOR_SYSTEM_PROMPT + f"\nStudent's Completed Courses: {completed_courses}"
    
    # Ensure the system prompt is always the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=context_prompt)] + messages
    else:
        messages[0] = SystemMessage(content=context_prompt)

    # Trim message history to prevent context window overflow
    trimmed = trim_messages(
        messages,
        max_tokens=20000,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
    )

    # Call the LLM with trimmed history
    response = llm_with_tools.invoke(trimmed)

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

# Compile the final graph
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