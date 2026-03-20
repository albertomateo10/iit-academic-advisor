import os
import json
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# 1. Load Environment Variables
load_dotenv()
ELASTIC_HOST = os.getenv("ELASTIC_HOST")
ELASTIC_USER = os.getenv("ELASTIC_USER")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

# --- PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for mappings
COURSE_MAPPING_PATH = os.path.join(BASE_DIR, "mappings", "course_mapping.json")
POLICY_MAPPING_PATH = os.path.join(BASE_DIR, "mappings", "policy_mapping.json")
PROGRAM_MAPPING_PATH = os.path.join(BASE_DIR, "mappings", "program_mapping.json") 

# Define absolute paths for scraped data
COURSES_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "scraped", "iit_courses.json")
POLICIES_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "scraped", "iit_policies.json")
PROGRAMS_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "scraped", "iit_programs.json") 

def load_json_schema(filepath):
    """Utility function to load external JSON mapping files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing schema file: {filepath}. Please ensure it exists.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {filepath}.")

def setup_elasticsearch():
    """Connects to ES and initializes the clean indices using external JSON schemas."""
    es = Elasticsearch(ELASTIC_HOST, basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD))
    
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch. Is Docker running?")
    
    # Load the external mappings
    course_mapping = load_json_schema(COURSE_MAPPING_PATH)
    policy_mapping = load_json_schema(POLICY_MAPPING_PATH)
    program_mapping = load_json_schema(PROGRAM_MAPPING_PATH) 
    
    # Reset indices for a clean ETL run
    for index_name in ["iit_courses", "iit_policies", "iit_programs"]: 
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"[-] Deleted old index: {index_name}")

    # Apply the dynamically loaded mappings
    es.indices.create(index="iit_courses", body=course_mapping)
    es.indices.create(index="iit_policies", body=policy_mapping)
    es.indices.create(index="iit_programs", body=program_mapping) 
    print("[+] Created fresh indices using external JSON schemas.")
    
    return es

def run_ingestion():
    es = setup_elasticsearch()

    print("\n[~] Loading BAAI/bge-large-en-v1.5 embedding model (this may take a moment)...")
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    print("[~] Initializing LangChain Semantic Chunker...")
    semantic_chunker = SemanticChunker(embedder)

    # --- PHASE 1: INGEST COURSES ---
    print("\n--- Starting Course Ingestion ---")
    with open(COURSES_DATA_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)

    for course in courses:
        desc_text = course.get("description", "")
        if not desc_text:
            desc_text = "No description available."
            
        vector = embedder.embed_query(desc_text)
        
        doc = {
            "course_id": course.get("course_id"),
            "course_name": course.get("course_name"),
            "description": desc_text,
            "credits": str(course.get("credits", "0")),
            "lecture_hours": course.get("lecture_hours", 0), 
            "lab_hours": course.get("lab_hours", 0),
            "prerequisites": course.get("prerequisites", []),
            "prerequisites_raw": course.get("prerequisites_raw", "None"),
            "description_vector": vector
        }
        
        es.index(index="iit_courses", document=doc)
    print(f"[+] Successfully embedded and indexed {len(courses)} courses.")

    # --- PHASE 2: INGEST POLICIES ---
    print("\n--- Starting Policy Ingestion ---")
    with open(POLICIES_DATA_PATH, "r", encoding="utf-8") as f:
        policies = json.load(f)

    total_policy_chunks = 0
    for policy in policies:
        tab_name = policy.get("tab_name", "Unknown")
        department = policy.get("department", "Unknown")
        url = policy.get("url", "Unknown")
        markdown_text = policy.get("content_markdown", "")
        
        if not markdown_text:
            continue

        print(f"  -> Semantic Chunking: [{department}] {tab_name}...")
        chunks = semantic_chunker.split_text(markdown_text)
        
        for i, chunk_text in enumerate(chunks):
            chunk_vector = embedder.embed_query(chunk_text)
            chunk_id = f"{department}_{tab_name}_chunk_{i}"
            
            doc = {
                "chunk_id": chunk_id,
                "document_type": policy.get("document_type", "academic_policy"),
                "department": department, 
                "tab_name": tab_name,
                "url": url,               
                "content_markdown": chunk_text,
                "content_vector": chunk_vector
            }
            
            es.index(index="iit_policies", document=doc)
            total_policy_chunks += 1

    print(f"[+] Successfully embedded and indexed {total_policy_chunks} policy chunks.")

    # --- PHASE 3: INGEST SPECIFIC PROGRAMS ---
    print("\n--- Starting Programs Ingestion ---")
    with open(PROGRAMS_DATA_PATH, "r", encoding="utf-8") as f:
        programs = json.load(f)

    total_prog_chunks = 0
    for prog in programs:
        prog_name = prog.get("program_name", "Unknown")
        tab_name = prog.get("tab_name", "Unknown")
        department = prog.get("department", "Unknown")
        level = prog.get("degree_level", "Unknown")
        url = prog.get("url", "Unknown")
        markdown_text = prog.get("content_markdown", "")
        
        if not markdown_text:
            continue

        print(f"  -> Semantic Chunking: [{level}] {prog_name} - {tab_name}...")
        
        # CRITICAL RAG STEP: Inject context so tables aren't orphaned during chunking!
        contextualized_text = f"Program: {prog_name} ({level}, {department} Department)\nSection: {tab_name}\n\n{markdown_text}"
        
        chunks = semantic_chunker.split_text(contextualized_text)
        
        for i, chunk_text in enumerate(chunks):
            chunk_vector = embedder.embed_query(chunk_text)
            
            # Format: ITM_Master_of_Cyber_Security_Requirements_chunk_1
            clean_name = prog_name.replace(" ", "_").replace("(", "").replace(")", "")
            chunk_id = f"{department}_{clean_name}_{tab_name}_chunk_{i}"
            
            doc = {
                "chunk_id": chunk_id,
                "document_type": prog.get("document_type", "specific_program"),
                "department": department,
                "degree_level": level,
                "program_name": prog_name,
                "tab_name": tab_name,
                "url": url,
                "content_markdown": chunk_text,
                "content_vector": chunk_vector
            }
            
            es.index(index="iit_programs", document=doc)
            total_prog_chunks += 1

    print(f"[+] Successfully embedded and indexed {total_prog_chunks} program chunks.")
    print("\n✅ Data Engineering Pipeline Complete! All indices are loaded and ready for LangGraph.")

if __name__ == "__main__":
    run_ingestion()