from typing import List
from typing_extensions import Optional

from langchain_core.tools import tool
from agents.retrievers import search_courses, search_policies, retrieve_program_info, check_course_eligibility


@tool
def search_iit_courses_tool(query: str, required_credits: Optional[str] = None) -> str:
    """
    Search the Illinois Institute of Technology (IIT) course catalog.

    Use this tool to look up ANY course code (e.g., "CS 584", "BIOL 503", "ITM 501"),
    course descriptions, credits, or prerequisites for ANY department at the university.
    DO NOT assume you only have access to CS or ITM courses. You have access to all of them.

    Use this tool to REPORT what a course's prerequisites are. Do NOT use it to determine
    whether a specific student is eligible to enroll — use check_course_eligibility_tool for that.

    Args:
        query (str): The natural language topic, course code, or concept to search for (e.g., "Cloud computing", "BIOL 503").
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


@tool
def check_course_eligibility_tool(course_ids: List[str], completed_courses: List[str]) -> str:
    """
    Deterministically checks whether the student is eligible to enroll in one or more courses,
    based on their exact AND/OR prerequisite logic and the student's completed courses.

    ALWAYS use this tool to determine ELIGIBLE / NOT ELIGIBLE — never work it out yourself from
    the raw prerequisite text. When you need to check several alternative or candidate courses in
    the same turn (for example, before recommending a course), pass ALL of their course codes in
    ONE call as a list, rather than calling this tool once per course.

    Args:
        course_ids (List[str]): The course codes to check, e.g. ["CS 584", "MATH 569", "CS 525"].
        completed_courses (List[str]): The student's completed courses, copied EXACTLY as given to
            you under "Student's Completed Courses" at the top of this conversation. Do not shorten,
            reformat, or guess this list.

    Returns:
        str: For each course, a deterministic ELIGIBLE / NOT ELIGIBLE verdict with the exact
        prerequisite reasoning. Trust this verdict completely and relay it to the student;
        do not override it with your own reading of the prerequisite text, and never state that
        completing one course unlocks another unless this tool confirmed it.
    """
    return check_course_eligibility(course_ids=course_ids, completed_courses=completed_courses)


agent_tools = [
    search_iit_courses_tool,
    search_iit_policies_tool,
    search_iit_programs_tool,
    check_course_eligibility_tool,
]