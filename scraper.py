import json, time, re
from playwright.sync_api import sync_playwright

BASE = "https://www.shl.com"
# The catalog redirects to /products/product-catalog/ — use that directly
CATALOG_BASE = "https://www.shl.com/products/product-catalog/"

def scrape_all_pages():
    all_products = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        
        # type=1 means Individual Test Solutions only (NOT pre-packaged job solutions)
        start = 0
        page_size = 12  # SHL shows 12 per page
        
        while True:
            url = f"{CATALOG_BASE}?action_doFilteringForm=Search&f=1&start={start}&type=1"
            print(f"Fetching page start={start} ...")
            
            page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait for the table to populate
            try:
                page.wait_for_selector("table tbody tr", timeout=15000)
            except:
                print(f"  No table rows found at start={start}, stopping.")
                break
            
            rows = page.query_selector_all("table tbody tr")
            if not rows:
                print("  No rows, done.")
                break
            
            found_on_page = 0
            for row in rows:
                try:
                    # Product name is in first <td> which contains an <a>
                    link_el = row.query_selector("td:first-child a")
                    if not link_el:
                        continue
                    
                    name = link_el.inner_text().strip()
                    href = link_el.get_attribute("href")
                    if not href:
                        continue
                    
                    # Make absolute URL
                    if href.startswith("/"):
                        full_url = BASE + href
                    else:
                        full_url = href
                    
                    # Get test type badges from the last <td>
                    # They appear as small letter badges: A, B, C, K, P, S, etc.
                    test_type_els = row.query_selector_all("td:last-child span, td:last-child .test-type, td span[class*='type']")
                    test_types = [el.inner_text().strip() for el in test_type_els if el.inner_text().strip()]
                    
                    # Fallback: get all text from last 2 tds
                    if not test_types:
                        tds = row.query_selector_all("td")
                        if len(tds) >= 2:
                            last_text = tds[-1].inner_text().strip()
                            # Test types are single letters
                            test_types = [c for c in last_text if c in "ABCDEKPS"]
                    
                    # Remote Testing and Adaptive flags (2nd and 3rd tds usually have checkmarks)
                    tds = row.query_selector_all("td")
                    remote_testing = False
                    adaptive = False
                    if len(tds) >= 3:
                        remote_text = tds[1].inner_text().strip()
                        adaptive_text = tds[2].inner_text().strip()
                        remote_testing = bool(remote_text) and remote_text != ""
                        adaptive = bool(adaptive_text) and adaptive_text != ""
                    
                    if name and full_url and "product-catalog" in full_url:
                        all_products.append({
                            "name": name,
                            "url": full_url,
                            "test_types": test_types,
                            "remote_testing": remote_testing,
                            "adaptive_irt": adaptive,
                            "description": ""
                        })
                        found_on_page += 1
                        
                except Exception as e:
                    print(f"  Row error: {e}")
                    continue
            
            print(f"  Found {found_on_page} products on this page.")
            
            if found_on_page == 0:
                break
            
            start += page_size
            time.sleep(1)
        
        # Now enrich each product with its description from its own page
        print(f"\nEnriching {len(all_products)} products with descriptions...")
        for i, product in enumerate(all_products):
            try:
                page.goto(product["url"], wait_until="domcontentloaded", timeout=30000)
                
                # Try meta description first (fastest)
                meta_desc = page.query_selector("meta[name='description']")
                desc = ""
                if meta_desc:
                    desc = meta_desc.get_attribute("content") or ""
                
                # Fallback: first meaningful paragraph
                if not desc or len(desc) < 20:
                    paras = page.query_selector_all("main p, .content p, article p")
                    texts = [p.inner_text().strip() for p in paras[:4] if len(p.inner_text().strip()) > 30]
                    desc = " ".join(texts)
                
                product["description"] = desc[:600]
                print(f"  [{i+1}/{len(all_products)}] {product['name']}")
                time.sleep(0.4)
                
            except Exception as e:
                print(f"  Failed to enrich {product['name']}: {e}")
        
        browser.close()
    
    return all_products


if __name__ == "__main__":
    products = scrape_all_pages()
    
    # Deduplicate by URL
    seen = set()
    unique = []
    for p in products:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    
    print(f"\n Done. {len(unique)} unique products saved to catalog.json")