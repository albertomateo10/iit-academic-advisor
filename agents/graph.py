import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Import the State and Tools we just built
from agents.state import AcademicAdvisorState
from agents.tools import (
    search_iit_courses_tool,
    search_iit_policies_tool,
    search_iit_programs_tool,
    check_course_eligibility_tool,
)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# 1. Initialize the LLM with Claude
llm = ChatAnthropic(
    # model="claude-sonnet-4-20250514",
    model="claude-haiku-4-5-20251001",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.1,
    max_tokens=2048
)

# 2. Bind the tools to the LLM so it knows what "Hands" it has
tools = [search_iit_courses_tool, search_iit_policies_tool, search_iit_programs_tool, check_course_eligibility_tool]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the System Prompt (Optimized for Haiku and natural conversation)
ADVISOR_SYSTEM_PROMPT = """You are an IIT ITM department academic advisor. Help students with courses, prerequisites, degree planning, and academic policies.

TOOL USAGE RULES:
- Use tools to look up courses, degree programs, or policies. Never guess course codes, credits, or prerequisites.
- You may call the courses tool multiple times within the same response if you need information about DIFFERENT courses (for example, checking the descriptions of several alternative courses). Call it once per distinct course you need to check.
- Do NOT call the tool again for a course you have already looked up in this turn, and do NOT call it again with a rephrased query for the same course. If a result does not contain the answer, tell the student you do not have that information in your catalog.
- If a question is about scheduling, calendars, tuition, or anything not in the course catalog or academic policies, say: "I don't have that information available. Please contact the ITM department or check the IIT website."

PREREQUISITES AND ENROLLMENT RULES (CRITICAL):
- When a student asks what a course's prerequisites ARE (without asking whether they personally qualify), report them explicitly by name and code, using the raw prerequisite text from search_iit_courses_tool. If a course has no prerequisites, explicitly say "This course has no prerequisites."
- When a student asks whether THEY can enroll in one or more specific courses (now or next semester), do NOT work out eligibility yourself by reading the AND/OR text. Instead, call check_course_eligibility_tool with the course code(s) and the student's completed courses (copy the list exactly from "Student's Completed Courses" below), and relay its ELIGIBLE / NOT ELIGIBLE verdict and reasoning. Trust that verdict completely; never override it with your own reading of the prerequisite text.
- If the tool confirms they meet all requirements, enthusiastically confirm they can enroll.
- If the tool says a requirement is not met, explain clearly what is missing, using the tool's own reasoning, in this style: "You cannot take this course yet because it requires [prerequisite]. Have you completed it?"

DEGREE PLANNING AND RECOMMENDATIONS:
- When a student asks about a specialization or degree program, use your tools to find the curriculum and provide a clear, well-organized response that separates the "Core/Required Courses" from the "Elective Courses".
- A student's completed course only counts toward a program's or certificate's required/elective credit total if that exact course appears in THAT SPECIFIC requirement's own course list. Before saying a course "counts" or subtracting it from the remaining credits/courses needed, check that it is literally listed as one of the required or elective options for that requirement. A completed course that does NOT appear in that list still has value (it may satisfy the PREREQUISITE of another course in the list), but it must never be counted as one of the required/elective courses itself, and must never be subtracted from the remaining credit total.
- When a student asks "Which course should I select next semester?", follow this exact process:
  1. Identify their specialization or degree, and use search_iit_programs_tool to find its full list of required and elective courses.
  2. List EVERY required or elective course they have NOT taken yet, checking each completed course strictly against that list (per the rule above) before treating it as already satisfied. If a requirement is phrased as "select N credits from the following list", include ALL courses in that list that are not yet completed, not just enough to reach N — you need the full candidate set before you can know which ones are actually eligible. If a requirement slot lists several alternative courses (e.g. "Course A or Course B"), include EVERY alternative too. Do not skip a course just because another one seems more obvious or you already have enough good options.
  3. Call check_course_eligibility_tool ONCE, passing the FULL list from step 2 together (every course, not a subset) and the student's completed courses (copy the list exactly from "Student's Completed Courses" below). Do not determine eligibility yourself, and do not call this tool separately for each course.
  4. Use the tool's verdicts, not your own judgment, to sort the courses into "eligible now" and "not yet eligible".
  5. Recommend ONLY from the courses the tool marked ELIGIBLE, and elaborate slightly on why they are good options for the student's academic path.
  6. Your response MUST include a section listing every course from step 2 that the tool marked NOT ELIGIBLE, each with the tool's stated reason, even if you already have enough eligible courses to recommend and even if the student didn't ask about them by name. Do not silently drop them from your answer. Never present a NOT ELIGIBLE course as equally valid alongside eligible ones.
  7. When you state how many courses or credits the student still needs, recompute it directly from the list in step 2, after applying the rule above. Never state a remaining-credits number without explicitly checking it against that list.
- Never claim that completing one course will make the student eligible for another course unless check_course_eligibility_tool has confirmed that second course's status this turn. Do not infer or guess chains between courses.

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
        max_tokens=30000,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
    )

    # Safety net: if a single turn produces a very large tool result (e.g. several full
    # program requirement pages), trim_messages combined with start_on="human" can end up
    # dropping every non-system message. Anthropic's API rejects a request with no
    # user/assistant messages at all, so guarantee the most recent human message onward
    # is always kept, regardless of the token budget.
    non_system = [m for m in trimmed if not isinstance(m, SystemMessage)]
    if not non_system:
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        if last_human_idx is not None:
            trimmed = [messages[0]] + messages[last_human_idx:]
        else:
            trimmed = messages

    # Call the LLM with trimmed history
    response = llm_with_tools.invoke(trimmed)

    return {"messages": [response]}

# We use LangGraph's prebuilt ToolNode to execute our Python search function
# ToolNode automatically supplies the InjectedState argument to check_course_eligibility_tool
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