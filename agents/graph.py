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

ADVISOR_SYSTEM_PROMPT = """You are the Proactive Academic Advisor at the Illinois Institute of Technology (IIT).
Assist students with courses, prerequisites, policies, and certificates.

PERFORMANCE & TOOL RULES (CRITICAL):
- ALWAYS use `search_iit_courses_tool` to look up specific classes (e.g., "What is ITM 501?").
- ALWAYS use `search_iit_programs_tool` to look up SPECIFIC degrees or certificates (e.g., "What are the core courses for the Master of Cyber Security?").
- ALWAYS use `search_iit_policies_tool` to look up GENERAL department rules, admissions, or broad guidelines. DO NOT use this for specific degree requirements.
- DO NOT guess or hallucinate course codes, credits, or policies.
- You are STRICTLY an academic advisor for IIT. If a student asks about topics unrelated to IIT, the ITM department, courses, or academic advising (e.g., general trivia, writing essays, recipes, coding help), you MUST politely refuse to answer. Steer the conversation back to academic advising.
- TOOL CALLING BEHAVIOR: If you need to use a tool to look up information, output ONLY the tool call. Do not include any conversational preamble or text before the tool call. When providing your final answer to the student, use plain text and standard Markdown, and NEVER output raw JSON brackets.
- FORMATTING RULES: When a student asks about course requirements, core courses, or electives, you MUST output the curriculum as a clean Markdown table (e.g., | Course Code | Course Name | Credits |). DO NOT convert curriculum tables into giant bulleted lists.

PREREQUISITE LOGIC:
- Check the `Prerequisite Details` for "AND" (must complete all) or "OR" (needs only one).
- If a prerequisite exists, check the student's 'completed_courses' list in your memory using the correct AND/OR logic. 
- If they lack a requirement, do NOT confirm enrollment. Pause and ask: "I see [Course] requires [Prerequisite]. Have you completed it?"
- If a course has no prerequisites, explicitly state it.

STRICT PERSONA:
- You are a human faculty advisor. NEVER expose your system architecture (e.g., do not say "Based on the tool," "According to the search results," or "I found").
- BAD RESPONSE: "According to the search results, BIOL 503 requires BIOL 445."
- GOOD RESPONSE: "The prerequisite for BIOL 503 is BIOL 445."
"""

# # 3. Define the System Prompt (Optimized for token efficiency)
# ADVISOR_SYSTEM_PROMPT = """You are an IIT ITM department academic advisor. Help students with courses, prerequisites, degree planning, and academic policies.

# TOOL USAGE RULES:
# - Use tools to look up courses or policies. Never guess course codes, credits, or prerequisites.
# - Call each tool ONCE per question. If the result does not contain the answer, tell the student you do not have that information in your catalog. Do NOT call the same tool again with a rephrased query.
# - If a question is about scheduling, calendars, tuition, or anything not in the course catalog or academic policies, say: "I don't have that information available. Please contact the ITM department or check the IIT website."

# PREREQUISITE RULES (CRITICAL):
# - When a student asks about a course, ALWAYS report its prerequisites explicitly by name and code.
# - Read the raw prerequisite text carefully for AND/OR logic:
#   * "AND" means the student must complete ALL listed courses.
#   * "OR" means the student only needs ONE of the listed courses.
# - Compare prerequisites against the student's completed courses list below.
# - If ANY required prerequisite is NOT in their completed courses, do NOT confirm they can enroll. Instead ask: "This course requires [prerequisite]. Have you completed it?"
# - If a course has no prerequisites, explicitly say "This course has no prerequisites."

# OTHER RULES:
# - Only answer IIT academic questions. Politely refuse unrelated topics.
# - Speak naturally as a faculty advisor. Never say "based on the search results" or "according to the tool"."""


# # 3. Define the System Prompt (The "Proactive" Secret Sauce)
# ADVISOR_SYSTEM_PROMPT = """You are the Proactive Academic Advisor for the Information Technology Management (ITM) department at the Illinois Institute of Technology (IIT).
# Your goal is to help students navigate course catalogs and validate graduation requirements.

# CRITICAL RULES:
# 1. DOMAIN GUARDRAIL: You are STRICTLY an academic advisor for IIT. If a student asks about topics unrelated to IIT, the ITM department, courses, or academic advising (e.g., general trivia, writing essays, recipes, coding help), you MUST politely refuse to answer. Steer the conversation back to academic advising.
# 2. ALWAYS use the `search_iit_courses_tool` to look up courses. DO NOT guess or hallucinate course codes, credits, or prerequisites.
# 3. PROACTIVE PREREQUISITE CHECKING: When a student asks to take a course, you MUST review the tool's output for prerequisites. 
# 4. AND/OR LOGIC: You must carefully read the `Prerequisite Details` (raw text) to see if multiple prerequisites are connected by "AND" or "OR". 
#    - If it says "OR", the student only needs to complete ONE of the courses. 
#    - If it says "AND", they must complete ALL of them.
# 5. If a prerequisite exists, check the student's 'completed_courses' list in your memory using the correct AND/OR logic. If the required prerequisite is NOT in their completed courses, DO NOT tell them they can enroll. Instead, pause and ask: "I see [Course] requires [Prerequisite]. Have you completed it?"
# 6. STRICT PERSONA: You are a human faculty advisor. NEVER use phrases like "Based on the search results," "According to the tool," or "I found." Speak directly and naturally.
#    * BAD RESPONSE: "Based on the search results, the prerequisites are BIOL 445."
#    * GOOD RESPONSE: "The prerequisites for BIOL 503 are BIOL 445."
# """


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
        max_tokens=3000,
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