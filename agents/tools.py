from typing_extensions import Optional

from langchain_core.tools import tool
# Import the raw Python function we just wrote and tested
from retrievers import search_courses 

@tool
def search_iit_courses_tool(query: str, required_credits: Optional[str] = None) -> str:
    """
    Search the Illinois Institute of Technology (IIT) course catalog.
    
    Use this tool WHENEVER a student asks about specific courses, topics of study, 
    or prerequisites. It performs a semantic search against the official catalog.
    
    Args:
        query (str): The natural language topic, course code, or concept to search for (e.g., "Cloud computing", "ITM 540").
        required_credits (str, optional): Use ONLY if the student specifically asks for a course with an exact number of credits (e.g., "3").
        
    Returns:
        str: A formatted string containing matching courses, their descriptions, credits, and REQUIRED PREREQUISITES. 
        PAY STRICT ATTENTION TO THE PREREQUISITES FIELD IN THE RETURNED TEXT.
    """
    # Simply pass the LLM's arguments down to your tested Elasticsearch function
    return search_courses(query=query, required_credits=required_credits)

# We define a list of tools that we will eventually bind to the LLM
agent_tools = [search_iit_courses_tool]