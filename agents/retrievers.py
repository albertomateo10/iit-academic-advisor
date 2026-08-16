import os
import re
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
ELASTIC_HOST = os.getenv("ELASTIC_HOST")
ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

# Connect to ES and load the embedding model
es = Elasticsearch(ELASTIC_HOST, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD))
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def search_courses(query: str, required_credits: str = None, top_k: int = 3):
    """
    Executes a Neuro-Symbolic Hybrid Search for IIT Courses.
    """
    # 1. Embed the student's query
    query_vector = embedder.embed_query(query)

    # 2. Define the k-NN (Vector) part
    knn_query = {
        "field": "description_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 50,
        "boost": 0.7 
    }

    # 3. Define the Symbolic/Keyword part
    text_query = {
        "bool": {
            # We use 'should' instead of 'must' so we don't accidentally exclude 
            # highly relevant semantic matches that might be missing an exact keyword
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["course_id^5","course_name^2", "description"],
                        "boost": 0.3 
                    }
                }
            ]
        }
    }

    # 4. Add Hard Symbolic Constraints (Exact Metadata Filtering)
    if required_credits:
        text_query["bool"]["filter"] = [
            {"term": {"credits": required_credits}}
        ]

    # 5. Execute the Query using Modern ES v8.x Syntax (No 'body' parameter)
    response = es.search(
        index="iit_courses", 
        knn=knn_query, 
        query=text_query, 
        size=top_k
    )

    # 6. Format the results
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        score = hit["_score"]
        course_info = (
            f"Course: {source.get('course_id')} - {source.get('course_name')}\n"
            f"Credits: {source.get('credits')} (Lecture Hours: {source.get('lecture_hours')}, Lab Hours: {source.get('lab_hours')})\n"            
            f"Required Course Codes: {', '.join(source.get('prerequisites', [])) or 'none'}\n"
            f"Prerequisite Details: {source.get('prerequisites_raw', 'none')}\n"
            f"Description: {source.get('description')}\n"
            f"Relevance Score: {score:.2f}\n"
        )
        results.append(course_info)

    return "\n---\n".join(results) if results else "No matching courses found."


def search_policies(query: str, department: Optional[str] = None, top_k: int = 2):
    """
    Executes a Hybrid Search for IIT academic policies, optionally filtered by department.
    """
    query_vector = embedder.embed_query(query)

    # --- 1. Vector Search (kNN) with optional filter ---
    knn_query = {
        "field": "content_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 50,
        "boost": 0.7,
    }
    # If the LLM specifies a department, force Elasticsearch to strictly filter for it
    if department:
        knn_query["filter"] = {"term": {"department": department}}

    # --- 2. Text Search (BM25) with optional filter ---
    text_query = {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["tab_name^10", "content_markdown^2"],
                        "boost": 0.3,
                    }
                }
            ]
        }
    }
    # Apply the same strict filter to the text search
    if department:
        text_query["bool"]["filter"] = [{"term": {"department": department}}]

    response = es.search(
        index="iit_policies",
        knn=knn_query,
        query=text_query,
        size=top_k,
    )

    # --- 3. Format Output for the LLM ---
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        dept_val = source.get('department', 'General')
        tab_val = source.get('tab_name', 'Policy')
        url_val = source.get('url', 'No URL available')
        content = source.get('content_markdown', '')
        
        # Inject context so the LLM knows exactly what department it is reading
        results.append(
            f"[{dept_val} Department - {tab_val}]\nSource URL: {url_val}\n{content}"
        )

    return "\n---\n".join(results) if results else f"No matching policies found for query: '{query}'."

def retrieve_program_info(query: str, department: Optional[str] = None, program_name: Optional[str] = None, top_k: int = 3) -> str:
    """Queries the iit_programs index for specific degree information."""

    # 1. Embed the search query
    query_vector = embedder.embed_query(query)

    # 2. department stays a hard filter, since it's always one of three exact,
    #    controlled strings ('Computer_Science', 'Information_Technology_Management',
    #    'Applied_Mathematics'), so an exact match is the correct behavior there.
    knn_query = {
        "field": "content_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 50,
    }
    if department:
        knn_query["filter"] = {"term": {"department": department}}

    # 3. program_name is now a SCORING BOOST, not a hard filter. Real program names
    #    always carry a suffix like "(with Computer Science)", so a caller passing a
    #    shortened or paraphrased name (e.g. "Master of Data Science") would previously
    #    get zero results instead of the closest matching program.
    text_query = None
    if program_name:
        text_query = {
            "bool": {
                "should": [
                    {"match": {"program_name": {"query": program_name, "boost": 3.0}}}
                ]
            }
        }
        if department:
            text_query["bool"]["filter"] = [{"term": {"department": department}}]

    try:
        if text_query:
            response = es.search(index="iit_programs", knn=knn_query, query=text_query, size=top_k)
        else:
            response = es.search(index="iit_programs", knn=knn_query, size=top_k)
        hits = response["hits"]["hits"]

        if not hits:
            return f"No specific program information found for query: '{query}'."

        # 4. Format the results cleanly for the LLM
        formatted_results = []
        for hit in hits:
            source = hit["_source"]
            doc = (
                f"Program: {source.get('program_name')} ({source.get('degree_level')})\n"
                f"Department: {source.get('department')}\n"
                f"Tab/Section: {source.get('tab_name')}\n"
                f"Source URL: {source.get('url')}\n"
                f"Content:\n{source.get('content_markdown')}\n"
            )
            formatted_results.append(doc)

        return "\n---\n".join(formatted_results)

    except Exception as e:
        return f"Database error while searching programs: {str(e)}"


# ============================================================
# DETERMINISTIC PREREQUISITE PARSER (Option A for eligibility)
# ============================================================
# Everything below evaluates AND/OR prerequisite logic in pure Python,
# instead of relying on the LLM to read and interpret the raw text.
# Tested against the AND/OR patterns actually present in iit_courses.json.

def _normalize_course_code(text: str) -> str:
    text = text.strip().upper()
    m = re.match(r'^([A-Z]+)\s*(\d+)', text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return text


def _strip_wrap(s: str) -> str:
    s = s.strip()
    if not (s.startswith('(') and s.endswith(')')):
        return s
    depth = 0
    for idx, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return s[1:-1].strip() if idx == len(s) - 1 else s
    return s


def _split_top_level(text: str, sep_word: str):
    """Split on a whole-word 'and'/'or' only when it sits outside any parentheses."""
    pattern = re.compile(r'\b' + sep_word + r'\b', re.IGNORECASE)
    parts, last = [], 0
    for m in pattern.finditer(text):
        depth = text[:m.start()].count('(') - text[:m.start()].count(')')
        if depth == 0:
            parts.append(text[last:m.start()].strip())
            last = m.end()
    parts.append(text[last:].strip())
    return parts


def _parse_expr(text: str):
    text = _strip_wrap(text.strip())
    and_parts = _split_top_level(text, 'and')
    if len(and_parts) > 1:
        return ('AND', [_parse_expr(p) for p in and_parts])
    or_parts = _split_top_level(text, 'or')
    if len(or_parts) > 1:
        return ('OR', [_parse_expr(p) for p in or_parts])
    text = _strip_wrap(text)
    if 'graduate standing' in text.lower():
        return ('GRAD_STANDING',)
    if not text:
        return None
    return ('COURSE', _normalize_course_code(text))


def _tokenize_prereq(raw: str):
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.lower() in ('none', 'no prerequisites', 'n/a'):
        return None
    text = re.sub(r'^Prerequisite\(s\):\s*', '', text, flags=re.IGNORECASE)
    text = re.split(r',\s*An asterisk', text)[0]
    text = re.sub(r'\bwith min\.\s*grade of [A-F][+-]?', '', text, flags=re.IGNORECASE)
    text = text.replace('*', '')
    text = text.strip().rstrip('.').strip()
    if not text:
        return None
    return _parse_expr(text)


def _evaluate(node, completed_set, grad_standing=True):
    if node is None:
        return True
    kind = node[0]
    if kind == 'GRAD_STANDING':
        return grad_standing
    if kind == 'COURSE':
        return node[1] in completed_set
    if kind == 'AND':
        return all(_evaluate(c, completed_set, grad_standing) for c in node[1])
    if kind == 'OR':
        return any(_evaluate(c, completed_set, grad_standing) for c in node[1])
    return False


def _describe(node, completed_set, grad_standing):
    kind = node[0]
    if kind == 'GRAD_STANDING':
        return f"Graduate standing ({'met' if grad_standing else 'not applicable'})"
    if kind == 'COURSE':
        code = node[1]
        return f"{code} ({'completed' if code in completed_set else 'NOT completed'})"
    joiner = ' AND ' if kind == 'AND' else ' OR '
    return "(" + joiner.join(_describe(c, completed_set, grad_standing) for c in node[1]) + ")"


def _explain(node, completed_set, grad_standing=True):
    if node is None:
        return "This course has no prerequisites."
    kind = node[0]
    if kind in ('COURSE', 'GRAD_STANDING'):
        return f"Requires {_describe(node, completed_set, grad_standing)}."
    if kind == 'AND':
        return f"Requires ALL of: {', '.join(_describe(c, completed_set, grad_standing) for c in node[1])}."
    if kind == 'OR':
        return f"Requires ONE of: {', '.join(_describe(c, completed_set, grad_standing) for c in node[1])}."
    return "Could not parse the prerequisite text automatically; verify manually."


def _fetch_course_by_id(course_id: str):
    """
    Looks up a single course by its course_id, reusing the same retrieval shape as
    search_courses (embedding + multi_match on course_id/course_name), just re-weighted
    to prioritize an exact code match. This reuses the same field access pattern that is
    already proven to work in search_courses, instead of a new, untested query shape.
    """
    normalized = _normalize_course_code(course_id)
    try:
        query_vector = embedder.embed_query(normalized)
        knn_query = {
            "field": "description_vector",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 50,
            "boost": 0.3,
        }
        text_query = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": normalized,
                            "fields": ["course_id^5", "course_name^2"],
                            "boost": 0.7,
                        }
                    }
                ]
            }
        }
        response = es.search(index="iit_courses", knn=knn_query, query=text_query, size=5)
        hits = response["hits"]["hits"]
        for hit in hits:
            if _normalize_course_code(hit["_source"].get("course_id", "")) == normalized:
                return hit["_source"]
        # Fallback: if no exact normalized match, return the top-ranked hit
        return hits[0]["_source"] if hits else None
    except Exception:
        return None


def check_course_eligibility(course_ids: list, completed_courses: list, grad_standing: bool = True) -> str:
    """
    Deterministically checks eligibility for one or more courses. No LLM reasoning involved:
    the AND/OR logic is parsed and evaluated entirely in Python against completed_courses.
    """
    completed_set = {_normalize_course_code(c) for c in (completed_courses or [])}
    lines = []
    for cid in course_ids:
        source = _fetch_course_by_id(cid)
        if not source:
            lines.append(f"{cid}: COULD NOT VERIFY - course not found in the catalog.")
            continue
        real_id = source.get("course_id", cid)
        name = source.get("course_name", "")
        tree = _tokenize_prereq(source.get("prerequisites_raw", ""))
        verdict = "ELIGIBLE" if _evaluate(tree, completed_set, grad_standing) else "NOT ELIGIBLE"
        lines.append(f"{real_id} - {name}: {verdict}. {_explain(tree, completed_set, grad_standing)}")
    return "\n".join(lines)


# if __name__ == "__main__":
#     print("Testing Hybrid Search...")
#     print("\n[Query: 'courses about protecting networks from hackers']")
#     print(search_courses("courses about protecting networks from hackers"))
    
#     # Example 2: Semantic + Symbolic Filter
#     print("\n[Query: 'courses about protecting networks', Filter: 3 credits]")
#     print(search_courses("courses about protecting networks", required_credits="3"))