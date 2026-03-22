import asyncio
import json
import os
import re
import random
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\u00a0', ' ').replace('\t', '')
    return re.sub(r'\n{3,}', '\n\n', text).strip()

async def safe_crawl_request(url: str, run_config: CrawlerRunConfig, crawler: AsyncWebCrawler, max_retries: int = 3):
    """
    SECURITY WRAPPER: Executes a crawl with randomized jitter and exponential backoff.
    """
    for attempt in range(max_retries):
        delay = random.uniform(3.2, 6.7)
        print(f"   [Anti-Ban] Sleeping for {delay:.2f}s...")
        await asyncio.sleep(delay)
        
        try:
            result = await crawler.arun(url=url, config=run_config)
            
            if result.success:
                return result
                
            error_msg = str(result.error_message)
            
            # --- THE FIX: DOM TIMEOUT CHECK ---
            # If the error specifically says it's waiting for a selector, the tab just doesn't exist.
            # Do NOT retry. Just return None and move on immediately.
            if "waiting for selector" in error_msg:
                print(f"   [-] Tab does not exist on this program page. Skipping gracefully.")
                return None
            # ----------------------------------
                
            if "429" in error_msg or "403" in error_msg or "Timeout" in error_msg:
                backoff_time = (2 ** attempt) * 5 + random.uniform(1.0, 3.0)
                print(f"   [!] Server throttling detected! Backing off for {backoff_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(backoff_time)
                continue 
                
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # --- THE FIX: Catching exceptions directly ---
            if "waiting for selector" in error_msg:
                print(f"   [-] Tab does not exist on this program page. Skipping gracefully.")
                return None
                
            if attempt == max_retries - 1:
                print(f"   [!] Max retries reached for {url}.")
                return None # Fail gracefully instead of crashing the whole script
            
            backoff_time = (2 ** attempt) * 5 + random.uniform(1.0, 3.0)
            print(f"   [!] Network error: {error_msg}. Retrying in {backoff_time:.2f}s...")
            await asyncio.sleep(backoff_time)
            
    return None

# --- PHASE 1: THE DISCOVERY SPIDER ---
# FIX 1: Changed base_url to target_url
async def discover_programs(department: str, target_url: str, container_id: str, level: str, crawler: AsyncWebCrawler):
    """Visits a department page, isolates a specific tab container, and extracts program links."""
    print(f"[*] Discovery Phase: Hunting for {level}s in {department}...")
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for=f"css:{container_id}", 
        css_selector=container_id
    )
    
    discovered_programs = []
    
    try:
        # FIX 2: Use target_url here
        result = await safe_crawl_request(url=target_url, run_config=run_config, crawler=crawler)
        
        if result.success and result.html:
            soup = BeautifulSoup(result.html, "html.parser")
            container = soup.select_one(container_id)
            
            if not container:
                print(f" [!] Could not find HTML container: {container_id}")
                return []

            links = container.find_all("a", href=True)
            
            for link in links:
                program_name = link.get_text(strip=True)
                relative_url = link['href']
                
                if not relative_url.startswith(("#", "javascript:")):
                    # FIX 3: Use target_url here to resolve the absolute path
                    absolute_url = urljoin(target_url, relative_url)
                    
                    if not any(p["url"] == absolute_url for p in discovered_programs):
                        discovered_programs.append({
                            "program_name": program_name,
                            "url": absolute_url,
                            "department": department,
                            "level": level
                        })
                        print(f"  -> Discovered: {program_name}")
                        
        return discovered_programs
    except Exception as e:
        print(f" [!] Failed to discover programs for {department}: {str(e)}")
        return []

# --- PHASE 2: THE EXTRACTION SPIDER ---
async def scrape_program_tab(url: str, tab_name: str, container_id: str, prog_data: dict, crawler: AsyncWebCrawler):
    """Visits a specific degree page and scrapes a specific tab (Overview, Requirements, etc)."""
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for=f"css:{container_id}", 
        css_selector=container_id,
        word_count_threshold=10,
        page_timeout=10000
    )

    try:
        result = await safe_crawl_request(url=url, run_config=run_config, crawler=crawler)
        if result.success:
            return {
                "document_type": "specific_program",
                "department": prog_data["department"],
                "degree_level": prog_data["level"],
                "program_name": prog_data["program_name"],
                "tab_name": tab_name,
                "url": url,
                "content_markdown": clean_extracted_text(result.markdown)
            }
        return None
    except Exception as e:
        return None

async def main():
    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    all_scraped_data = []
    
    departments_to_scan = [
        {
            "department": "Computer_Science",
            # We use the clean base URL without the #hash tags here
            "base_url": "https://catalog.iit.edu/graduate/colleges/computing/computer-science/",
            # The CSS IDs you provided
            "masters_container": "#degreeprogramstextcontainer",
            "certs_container": "#certificatestextcontainer"
        },
        {
            "department": "Computer_Science",
            # We use the clean base URL without the #hash tags here
            "base_url": "https://catalog.iit.edu/graduate/colleges/computing/applied-mathematics/",
            # The CSS IDs you provided
            "masters_container": "#degreeprogramstextcontainer",
            "certs_container": "#certificatestextcontainer"
        },
        {
            "department": "Information_Technology_Management",
            # We use the clean base URL without the #hash tags here
            "base_url": "https://catalog.iit.edu/graduate/colleges/computing/information-technology-management/",
            # The CSS IDs you provided
            "masters_container": "#degreeprogramstextcontainer",
            "certs_container": "#certificatestextcontainer"
        }
    ]

    async with AsyncWebCrawler(config=browser_config) as crawler:
        
# --- EXECUTE PHASE 1 ---
        all_discovered_links = []
        for dept in departments_to_scan:
            # FIX 4: Dynamically construct the correct URL with the hash so the tab is VISIBLE!
            masters_url = dept["base_url"] + dept["masters_container"].replace("container", "")
            certs_url = dept["base_url"] + dept["certs_container"].replace("container", "")
            
            masters = await discover_programs(dept["department"], masters_url, dept["masters_container"], "Master", crawler)
            all_discovered_links.extend(masters)
            await asyncio.sleep(random.uniform(2, 4))
            
            certs = await discover_programs(dept["department"], certs_url, dept["certs_container"], "Certificate", crawler)
            all_discovered_links.extend(certs)
            await asyncio.sleep(random.uniform(2, 4))
        print(f"\n[+] Discovery Complete! Found {len(all_discovered_links)} total programs to scrape.\n")
        
        # --- EXECUTE PHASE 2 ---
        # THE FIX: We now use a LIST of possible IDs for each tab to handle inconsistent departments!
        target_tabs = {
            "Overview": ["#textcontainer", "#overviewtextcontainer"], 
            "Program Requirements": ["#programrequirementstextcontainer"],
            "Specializations": ["#specializationstextcontainer"]
        }

        print("[*] Extraction Phase: Scraping individual program pages...")
        for i, prog in enumerate(all_discovered_links):
            print(f"Processing [{i+1}/{len(all_discovered_links)}]: {prog['program_name']}")
            
            for tab_name, container_ids in target_tabs.items():
                
                # --- URL NORMALIZATION ---
                # Ensure the program URL ends with a slash so the server doesn't force a redirect
                clean_prog_url = prog["url"]
                if not clean_prog_url.endswith("/"):
                    clean_prog_url += "/"
                
                # --- THE FALLBACK LOOP ---
                # Loop through our list of possible IDs (e.g., try #textcontainer, then try #overviewtextcontainer)
                for container_id in container_ids:
                    # Safely attach the hash (e.g., #text or #overviewtext)
                    tab_url = clean_prog_url + container_id.replace("container", "")
                    
                    doc = await scrape_program_tab(tab_url, tab_name, container_id, prog, crawler)
                    
                    if doc:
                        all_scraped_data.append(doc)
                        print(f"  -> Extracted '{tab_name}'")
                        break # SUCCESS! Stop trying fallbacks and move to the next tab
                # -------------------------
                    
            await asyncio.sleep(random.uniform(3.0, 5.0))

    # Save to JSON
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "scraped", "dynamic_programs.json"))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_scraped_data, f, indent=4)
        
    print(f"\n✅ SUCCESS: Spider finished! Saved {len(all_scraped_data)} data chunks to dynamic_programs.json")

if __name__ == "__main__":
    asyncio.run(main())