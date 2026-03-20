from typing_extensions import Optional

from langchain_core.tools import tool
from agents.retrievers import search_courses, search_policies, retrieve_program_info


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
    return search_courses(query=query, required_credits=required_credits)


@tool
def search_iit_policies_tool(query: str, department: str = None) -> str:
    """
    Search for GENERAL IIT academic policies at the department level.
    
    Call this tool when the user asks about:
    - General department admission requirements (e.g., "Do I need the GRE for Computer Science?", "What is the minimum GPA?")
    - General overview of a department's goals or mission.
    - Department-wide rules or policies.

    CRITICAL ROUTING INSTRUCTION:
    DO NOT use this tool if the user is asking about a SPECIFIC Master's degree or Certificate (e.g., "What are the core courses for the Master of Cyber Security?"). For specific degrees, use the 'search_iit_programs_tool' instead!

    Args:
        query: The topic to search (e.g., "GRE requirement", "minimum GPA for admission").
        department: (Optional) The specific department. If specified, MUST be exactly 'Information_Technology_Management' or 'Computer_Science'.
    """
    return search_policies(query=query, department=department)


@tool
def search_iit_programs_tool(query: str, department: str = None, program_name: str = None) -> str:
    """
    Use this tool to find information about SPECIFIC Master's degrees and Certificates.
    
    Call this tool when the user asks about:
    - Core courses or requirements for a specific degree (e.g., "What courses do I need for a Master of Cyber Security?")
    - Specializations available within a specific program.
    - Credit hour requirements for a specific certificate.
    - Admission requirements for a specific Master's program.
    
    DO NOT use this tool for general university policies, academic probation, or general grading rules.
    
    Args:
        query: The specific question about the program (e.g., "core courses and electives").
        department: (Optional) The department name. Use EXACTLY 'Information_Technology_Management' or 'Computer_Science'.
        program_name: (Optional) The specific name of the degree (e.g., "Master of Cyber Forensics and Security").
    """
    return retrieve_program_info(query=query, department=department, program_name=program_name)


agent_tools = [search_iit_courses_tool, search_iit_policies_tool, search_iit_programs_tool]