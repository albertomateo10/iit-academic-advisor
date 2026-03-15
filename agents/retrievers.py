import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
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
            f"Credits: {source.get('credits')}\n"
            f"Prerequisite Course Codes: {', '.join(source.get('prerequisites', [])) or 'None'}\n"
            f"Prerequisite Details: {source.get('prerequisites_raw', 'None')}\n"
            f"Description: {source.get('description')}\n"
            f"Relevance Score: {score:.2f}\n"
        )
        results.append(course_info)

    return "\n---\n".join(results) if results else "No matching courses found."

# if __name__ == "__main__":
#     print("Testing Hybrid Search...")
#     print("\n[Query: 'courses about protecting networks from hackers']")
#     print(search_courses("courses about protecting networks from hackers"))
    
#     # Example 2: Semantic + Symbolic Filter
#     print("\n[Query: 'courses about protecting networks', Filter: 3 credits]")
#     print(search_courses("courses about protecting networks", required_credits="3"))