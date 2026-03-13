import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def scrape_itm_policy_tab(url: str, tab_name: str, container_id: str):
    """
    Scrapes a specific ITM department tab and extracts ONLY that container's Markdown.
    """
    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        # FIX 1: Dynamically wait for the correct tab container to load
        wait_for=f"css:{container_id}", 
        # FIX 2: Target ONLY this specific container to get clean Markdown (no nav menus)
        css_selector=container_id,
        word_count_threshold=10
    )

    print(f"Scraping ITM Policy: {tab_name}...")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success:
            policy_doc = {
                "document_type": "academic_policy",
                "tab_name": tab_name,
                "url": url,
                # .strip() removes any accidental leading/trailing whitespace
                "content_markdown": result.markdown.strip(), 
                "raw_html": result.html 
            }
            return policy_doc
        else:
            print(f"Failed to scrape {tab_name}: {result.error_message}")
            return None

async def main():
    # We map the URL to the specific CSS container ID for that exact tab
    itm_tabs = {
        "Overview": {
            "url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#text",
            "id": "#textcontainer"
        },
        "Admissions": {
            "url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#admissionstext",
            "id": "#admissionstextcontainer"
        },
        "Degree_Programs": {
            "url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#degreeprogramstext",
            "id": "#degreeprogramstextcontainer"
        },
        "Certificates": {
            "url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/#certificatestext",
            "id": "#certificatestextcontainer"
        }
    }
    
    all_policies = []
    
    # Unpack the dictionary to pass both the URL and the container ID
    for name, data in itm_tabs.items():
        doc = await scrape_itm_policy_tab(data["url"], name, data["id"])
        
        if doc:
            all_policies.append(doc)
            print(f"  -> Success: Extracted {len(doc['content_markdown'])} characters of clean Markdown.")
        
        # Keep it polite!
        await asyncio.sleep(4)


    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Join the directory path with the filename
    file_path = os.path.abspath(os.path.join(script_dir, "..", "data", "scraped", "itm_policies.json"))
    # Save to a separate JSON file for inspection
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_policies, f, indent=4)
    
    print(f"\nSaved {len(all_policies)} policy documents to itm_policies.json")

if __name__ == "__main__":
    asyncio.run(main())