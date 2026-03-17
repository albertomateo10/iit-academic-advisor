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
    """Parses raw text into strict formats, capturing labs, lectures, variable credits, and cleaning typography."""
    
    # --- INTERNAL HELPER: Advanced Text Cleaning ---
    def advanced_clean(text):
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Replace Unicode non-breaking spaces (\u00a0 or \xa0) and line breaks (\n, \r)
        text = text.replace('\xa0', ' ').replace('\u00a0', ' ')
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 2. Add space between lowercase and uppercase letters (e.g., "bothMATH" -> "both MATH")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # 3. Add space between digits and letters (e.g., "611with" -> "611 with")
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # 4. Add space between letters and digits (e.g., "MATH522" -> "MATH 522")
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        
        # 5. Normalize multiple spaces into a single one and trim whitespace from both ends
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    cleaned_data = []
    if not raw_data: 
        return cleaned_data
        
    for course in raw_data:
        # Apply the advanced cleaning helper to all extracted fields
        code = advanced_clean(course.get("course_code", ""))
        title = advanced_clean(course.get("course_title", ""))
        desc = advanced_clean(course.get("description", ""))
        prereqs_raw = advanced_clean(course.get("prereqs_raw", ""))
        credits_raw = advanced_clean(course.get("credits", ""))
        
        # --- PARSE LECTURE, LAB, AND CREDITS ---
        
        # 1. Parse Lecture Hours (Defaults to 0 if not found)
        lecture_match = re.search(r'Lecture:\s*(\d+)', credits_raw, re.IGNORECASE)
        lecture_hours = int(lecture_match.group(1)) if lecture_match else 0
        
        # 2. Parse Lab Hours (Defaults to 0 if not found)
        lab_match = re.search(r'Lab:\s*(\d+)', credits_raw, re.IGNORECASE)
        lab_hours = int(lab_match.group(1)) if lab_match else 0
        
        # 3. Parse Credits (Handles both integers and "Variable" string)
        credit_match = re.search(r'Credit(?:s)?:\s*(\d+|Variable)', credits_raw, re.IGNORECASE)
        
        if credit_match:
            extracted_credit = credit_match.group(1)
            # If numeric, convert to int; if "Variable", capitalize correctly
            final_credits = int(extracted_credit) if extracted_credit.isdigit() else extracted_credit.title()
        else:
            final_credits = 0 # Safety fallback
            
        # --- PARSE PREREQUISITES ---
        # Since prereqs_raw is now cleaned and correctly spaced (e.g., "MATH 101"), 
        # the regex pattern becomes more reliable.
        prereqs_list = re.findall(r'[A-Z]{3,4}\s\d{3,4}', prereqs_raw)
        unique_prereqs = list(set(prereqs_list))

        cleaned_data.append({
            "course_id": code,
            "course_name": title,
            "description": desc,
            "lecture_hours": lecture_hours,
            "lab_hours": lab_hours,
            "credits": final_credits,
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
                delay = random.uniform(2.5, 5.0)
                print(f"Sleeping for {delay:.2f} seconds to avoid rate limiting...\n")
                await asyncio.sleep(delay)
                
    return all_courses

async def main():
    # Example list of department URLs
    department_urls = [
        "https://catalog.iit.edu/graduate/courses/arch/",
        "https://catalog.iit.edu/graduate/courses/biol/",
        "https://catalog.iit.edu/graduate/courses/bme/",
        "https://catalog.iit.edu/graduate/courses/bus/",
        "https://catalog.iit.edu/graduate/courses/che/",
        "https://catalog.iit.edu/graduate/courses/chem/",
        "https://catalog.iit.edu/graduate/courses/cae/",
        "https://catalog.iit.edu/graduate/courses/com/",
        "https://catalog.iit.edu/graduate/courses/cs/",
        "https://catalog.iit.edu/graduate/courses/csp/",
        "https://catalog.iit.edu/graduate/courses/ece/",
        "https://catalog.iit.edu/graduate/courses/enve/",
        "https://catalog.iit.edu/graduate/courses/ems/",
        "https://catalog.iit.edu/graduate/courses/fdsn/",
        "https://catalog.iit.edu/graduate/courses/engr/",
        "https://catalog.iit.edu/graduate/courses/hist/",
        "https://catalog.iit.edu/graduate/courses/hum/",
        "https://catalog.iit.edu/graduate/courses/intm/",
        "https://catalog.iit.edu/graduate/courses/idn/",
        "https://catalog.iit.edu/graduate/courses/idx/",
        "https://catalog.iit.edu/graduate/courses/itmd/",
        "https://catalog.iit.edu/graduate/courses/itmm/",
        "https://catalog.iit.edu/graduate/courses/itmo/",
        "https://catalog.iit.edu/graduate/courses/itms/",
        "https://catalog.iit.edu/graduate/courses/itmt/",
        "https://catalog.iit.edu/graduate/courses/la/",
        "https://catalog.iit.edu/graduate/courses/msc/",
        "https://catalog.iit.edu/graduate/courses/max/",
        "https://catalog.iit.edu/graduate/courses/msf/",
        "https://catalog.iit.edu/graduate/courses/math/",
        "https://catalog.iit.edu/graduate/courses/mba/",
        "https://catalog.iit.edu/graduate/courses/mmae/",
        "https://catalog.iit.edu/graduate/courses/phil/",
        "https://catalog.iit.edu/graduate/courses/phys/",
        "https://catalog.iit.edu/graduate/courses/psyc/",
        "https://catalog.iit.edu/graduate/courses/pa/",
        "https://catalog.iit.edu/graduate/courses/sci/",
        "https://catalog.iit.edu/graduate/courses/sens/",
        "https://catalog.iit.edu/graduate/courses/ssci/",
        "https://catalog.iit.edu/graduate/courses/stat/",
        "https://catalog.iit.edu/graduate/courses/ssb/",
        "https://catalog.iit.edu/graduate/courses/sam/",
        "https://catalog.iit.edu/graduate/courses/smgt/",
        "https://catalog.iit.edu/graduate/courses/tech/"
    ]
    
    # 1. Run the safe ETL process
    final_data = await safe_catalog_etl(department_urls)
    
    # 2. Save the results to a file for inspection
    if final_data:
        save_to_json(final_data, "iit_courses.json")

if __name__ == "__main__":
    asyncio.run(main())