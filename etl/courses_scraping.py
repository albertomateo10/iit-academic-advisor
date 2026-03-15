import asyncio
import json
import random
import os
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import re

# 1. The Schema
course_schema = {
    "name": "IIT_Course_Extractor",
    "baseSelector": "div.courseblock", 
    "fields": [
        {"name": "course_code", "selector": "div.coursecode", "type": "text"},
        {"name": "course_title", "selector": "div.coursetitle", "type": "text"},
        {"name": "description", "selector": "div.courseblockdesc p.noindent", "type": "text"},
        {"name": "prereqs_raw", "selector": "div.courseblockattr:not(.hours)", "type": "text"},
        {"name": "credits", "selector": "div.courseblockattr.hours", "type": "text"}
    ]
}

async def scrape_courses(url: str, crawler: AsyncWebCrawler):
    """Scrapes structured course data using an active crawler instance."""
    extraction_strategy = JsonCssExtractionStrategy(schema=course_schema)
    run_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode=CacheMode.BYPASS,
        wait_for="css:div.sc_sccoursedescs" 
    )

    print(f"Executing CSS extraction on {url}...")
    result = await crawler.arun(url=url, config=run_config)
        
    if result.success:
        extracted_data = json.loads(result.extracted_content)
        print(f"  -> Extracted {len(extracted_data)} courses.")
        return extracted_data
    else:
        print(f"  -> Crawl failed: {result.error_message}")
        return None

def clean_course_data(raw_data):
    """Parses raw text into strict formats, capturing labs, lectures, and variable credits."""
    cleaned_data = []
    if not raw_data: 
        return cleaned_data
        
    for course in raw_data:
        code = course.get("course_code", "").replace("\xa0", " ").strip()
        title = course.get("course_title", "").strip()
        desc = course.get("description", "").strip()
        
        # --- NEW: PARSE LECTURE, LAB, AND CREDITS ---
        # The raw string might be "LECTURE: 3 LAB: 0 CREDITS: 3" or "CREDIT: Variable"
        credits_raw = course.get("credits", "").strip()
        
        # 1. Parse Lecture Hours (Default to 0 if not found)
        lecture_match = re.search(r'Lecture:\s*(\d+)', credits_raw, re.IGNORECASE)
        lecture_hours = int(lecture_match.group(1)) if lecture_match else 0
        
        # 2. Parse Lab Hours (Default to 0 if not found)
        lab_match = re.search(r'Lab:\s*(\d+)', credits_raw, re.IGNORECASE)
        lab_hours = int(lab_match.group(1)) if lab_match else 0
        
        # 3. Parse Credits (Handles both integers and "Variable")
        # The regex looks for "Credit:" or "Credits:" followed by digits OR the word "Variable"
        credit_match = re.search(r'Credit(?:s)?:\s*(\d+|Variable)', credits_raw, re.IGNORECASE)
        
        if credit_match:
            extracted_credit = credit_match.group(1)
            # If it is a number, convert to int. If it is "Variable", keep the string "Variable"
            final_credits = int(extracted_credit) if extracted_credit.isdigit() else extracted_credit.title()
        else:
            final_credits = 0 # Absolute fallback
            
        # Parse Prerequisites...
        prereqs_raw = course.get("prereqs_raw", "").replace("\xa0", " ").strip()
        prereqs_list = re.findall(r'[A-Z]{3,4}\s\d{3,4}', prereqs_raw)
        unique_prereqs = list(set(prereqs_list))

        cleaned_data.append({
            "course_id": code,
            "course_name": title,
            "description": desc,
            "lecture_hours": lecture_hours,
            "lab_hours": lab_hours,
            "credits": final_credits,             # Now holds 3, 1, or "Variable"
            "prerequisites": unique_prereqs,
            "prerequisites_raw": prereqs_raw
        })
        
    return cleaned_data

def save_to_json(data: list, filename: str = "iit_courses.json"):
    """
    Saves the extracted Python list of dictionaries to a local JSON file
    in the same directory where this script is located.
    """
    # 1. Get the absolute path of the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Join the script directory with the desired filename
    file_path = os.path.abspath(os.path.join(script_dir, "..", "data", "scraped", filename))

    # 3. Ensure we use utf-8 encoding so special characters are preserved
    with open(file_path, "w", encoding="utf-8") as f:
        # indent=4 makes the JSON easily readable for human inspection
        json.dump(data, f, indent=4) 
    
    print(f"\n[+] Data successfully saved to: {file_path}")

async def safe_catalog_etl(urls: list[str]):
    """Iterates through URLs safely with asyncio.sleep()."""
    
    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    all_courses = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, url in enumerate(urls):
            print(f"--- Processing [{index + 1}/{len(urls)}] ---")
            
            raw_courses = await scrape_courses(url, crawler)
            cleaned_courses = clean_course_data(raw_courses)
            all_courses.extend(cleaned_courses)
            
            if index < len(urls) - 1:
                delay = random.uniform(3.5, 5.0)
                print(f"Sleeping for {delay:.2f} seconds to avoid rate limiting...\n")
                await asyncio.sleep(delay)
                
    return all_courses

async def main():
    # Example list of department URLs
    department_urls = [
        "https://catalog.iit.edu/graduate/courses/arch/",
        "https://catalog.iit.edu/graduate/courses/biol/"  # Commented out to keep the test quick
    ]
    
    # 1. Run the safe ETL process
    final_data = await safe_catalog_etl(department_urls)
    
    # 2. Save the results to a file for inspection
    if final_data:
        save_to_json(final_data, "iit_courses.json")

if __name__ == "__main__":
    asyncio.run(main())