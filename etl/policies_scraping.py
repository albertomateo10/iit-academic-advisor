import asyncio
import json
import os
import re
import random
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\u00a0', ' ')
    text = text.replace('\t', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# FIX 1: Pass the department name into the scraping function
async def scrape_policy_tab(url: str, tab_name: str, container_id: str, department: str):
    """
    Scrapes a specific department tab and extracts ONLY that container's Markdown.
    """
    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    
    # We use basic Markdown extraction. Crawl4AI handles HTML tables automatically!
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for=f"css:{container_id}", 
        css_selector=container_id,
        word_count_threshold=10
    )

    print(f"Scraping [{department}] Policy: {tab_name}...")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success:
            clean_markdown = clean_extracted_text(result.markdown)
            clean_html = clean_extracted_text(result.html)
            clean_html = re.sub(r'\n{2,}', '\n', clean_html) 

            policy_doc = {
                "document_type": "academic_policy",
                "department": department, 
                "tab_name": tab_name,
                "url": url,
                "content_markdown": clean_markdown, 
                "raw_html": clean_html 
            }
            return policy_doc
        else:
            print(f"Failed to scrape {department} - {tab_name}: {result.error_message}")
            return None

async def main():
    # FIX 3: Nested dictionary to handle multiple departments
    catalog_urls = {
        "ITM": {
            "Overview": {"url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#text", "id": "#textcontainer"},
            "Admissions": {"url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#admissionstext", "id": "#admissionstextcontainer"},
            "Degree_Programs": {"url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#degreeprogramstext", "id": "#degreeprogramstextcontainer"},
            "Certificates": {"url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#certificatestext", "id": "#certificatestextcontainer"}
        },
        "Computer_Science": {
            "Overview": {"url": "https://catalog.iit.edu/graduate/colleges/computing/computer-science/#text", "id": "#textcontainer"},
            "Admissions": {"url": "https://catalog.iit.edu/graduate/colleges/computing/computer-science/#admissionsrequirementstext", "id": "#admissionsrequirementstextcontainer"},
            "Degree_Programs": {"url": "https://catalog.iit.edu/graduate/colleges/computing/computer-science/#degreeprogramstext", "id": "#degreeprogramstextcontainer"},
            "Certificates": {"url": "https://catalog.iit.edu/graduate/colleges/computing/computer-science/#certificatestext", "id": "#certificatestextcontainer"}
        },
        "Applied_Mathematics": {
            "Overview": {"url": "https://catalog.iit.edu/graduate/colleges/computing/applied-mathematics/", "id": "#textcontainer"},
            "Admissions": {"url": "https://catalog.iit.edu/graduate/colleges/computing/applied-mathematics/#admissionstext", "id": "#admissionstextcontainer"},
            "Degree_Programs": {"url": "https://catalog.iit.edu/graduate/colleges/computing/applied-mathematics/#degreeprogramstext", "id": "#degreeprogramstextcontainer"}
        }
    }
    
    all_policies = []
    
    total_tasks = sum(len(tabs) for tabs in catalog_urls.values())
    current_task = 0
    
    for department_name, tabs in catalog_urls.items():
        for tab_name, data in tabs.items():
            current_task += 1
            
            doc = await scrape_policy_tab(data["url"], tab_name, data["id"], department_name)
            
            if doc:
                all_policies.append(doc)
                print(f"  -> Success: Extracted {len(doc['content_markdown'])} chars.")
            
            # 4. Apply the random sleep jitter, skipping the very last iteration
            if current_task < total_tasks:
                delay = random.uniform(2.5, 5.0)
                print(f"Sleeping for {delay:.2f} seconds to avoid rate limiting...\n")
                await asyncio.sleep(delay)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(script_dir, "..", "data", "scraped", "all_policies.json"))
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_policies, f, indent=4)
    
    print(f"\nSaved {len(all_policies)} policy documents to all_policies.json")

if __name__ == "__main__":
    asyncio.run(main())