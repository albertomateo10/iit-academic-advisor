from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AcademicAdvisorState(TypedDict):
    """
    Represents the complete state of the conversation and the student's context.
    This state is passed to every node in the LangGraph execution.
    """
    # 1. Conversation History
    # The 'add_messages' reducer ensures that new messages are APPENDED to the 
    # list, rather than overwriting the entire list every turn.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 2. Student Academic Context (The "Proactive" Enablers)
    # We store the courses the student explicitly tells us they have passed.
    completed_courses: list[str]
    
    # Optional: Track GPA if you want to enforce rules like "requires 3.0 GPA"
    current_gpa: float 
    
    # We can use a flag to track if the agent is currently waiting for the 
    # student to answer a specific question (e.g., "Did you take ITM 311?")
    awaiting_prereq_confirmation: bool