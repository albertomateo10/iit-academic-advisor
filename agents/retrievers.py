import os
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

def retrieve_program_info(query: str, department: Optional[str] = None, program_name: Optional[str] = None, top_k: int = 5) -> str:
    """Queries the iit_programs index for specific degree information."""
    
    # 1. Embed the search query
    query_vector = embedder.embed_query(query)
    
    # 2. Build exact-match filters (The beauty of your 'keyword' schema!)
    filter_clauses = []
    if department:
        filter_clauses.append({"term": {"department": department}})
    if program_name:
        # Using a match query here in case the LLM slightly misspells the program name
        filter_clauses.append({"match": {"program_name": program_name}})
        
    # 3. Construct the k-NN vector search query
    es_query = {
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 50
        }
    }
    
    # Apply filters if the LLM provided any
    if filter_clauses:
        es_query["knn"]["filter"] = {
            "bool": {
                "must": filter_clauses
            }
        }
        
    try:
        response = es.search(index="iit_programs", body=es_query)
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

# if __name__ == "__main__":
#     print("Testing Hybrid Search...")
#     print("\n[Query: 'courses about protecting networks from hackers']")
#     print(search_courses("courses about protecting networks from hackers"))
    
#     # Example 2: Semantic + Symbolic Filter
#     print("\n[Query: 'courses about protecting networks', Filter: 3 credits]")
#     print(search_courses("courses about protecting networks", required_credits="3"))