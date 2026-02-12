"""
CrossRef API module for fetching paper metadata from DOIs.
"""

import requests
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


CROSSREF_API = "https://api.crossref.org/works/"
# Be polite to CrossRef API
RATE_LIMIT_DELAY = 0.1  # seconds between requests


def fetch_metadata(doi: str, email: Optional[str] = None) -> Optional[dict]:
    """
    Fetch metadata for a single DOI from CrossRef.
    
    Args:
        doi: The DOI to look up
        email: Optional email for polite pool (faster rate limits)
    
    Returns:
        Dictionary with paper metadata or None if not found
    """
    headers = {"User-Agent": f"PaperSearch/1.0 (mailto:{email})" if email else "PaperSearch/1.0"}
    
    try:
        # Clean up DOI - remove common prefixes
        clean_doi = doi.strip()
        for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
            if clean_doi.lower().startswith(prefix.lower()):
                clean_doi = clean_doi[len(prefix):]
        
        url = f"{CROSSREF_API}{clean_doi}"
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            return None
        
        work = data["message"]
        
        # Extract authors
        authors = []
        for author in work.get("author", []):
            if "given" in author and "family" in author:
                authors.append(f"{author['given']} {author['family']}")
            elif "family" in author:
                authors.append(author['family'])
            elif "name" in author:
                authors.append(author['name'])
        
        # Extract year
        year = None
        if "published-print" in work:
            date_parts = work["published-print"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
        elif "published-online" in work:
            date_parts = work["published-online"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
        elif "created" in work:
            date_parts = work["created"].get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
        
        # Extract journal/container title
        journal = None
        container = work.get("container-title", [])
        if container:
            journal = container[0]
        
        # Extract abstract (may contain HTML/XML)
        abstract = work.get("abstract", "")
        if abstract:
            # Basic cleanup of JATS XML tags
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = abstract.strip()
        
        # Extract keywords/subjects
        keywords = work.get("subject", [])
        
        return {
            "doi": clean_doi,
            "title": work.get("title", [""])[0] if work.get("title") else None,
            "authors": authors,
            "abstract": abstract if abstract else None,
            "journal": journal,
            "year": year,
            "keywords": keywords
        }
        
    except requests.RequestException as e:
        print(f"Error fetching {doi}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {doi}: {e}")
        return None


def fetch_metadata_batch(
    dois: list[str],
    email: Optional[str] = None,
    max_workers: int = 5,
    progress: bool = True
) -> dict[str, Optional[dict]]:
    """
    Fetch metadata for multiple DOIs with rate limiting.
    
    Args:
        dois: List of DOIs to look up
        email: Optional email for polite pool
        max_workers: Number of concurrent requests (keep low to be nice to API)
        progress: Show progress bar
    
    Returns:
        Dictionary mapping DOI -> metadata (or None if not found)
    """
    results = {}
    
    def fetch_with_delay(doi):
        time.sleep(RATE_LIMIT_DELAY)
        return doi, fetch_metadata(doi, email)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_with_delay, doi): doi for doi in dois}
        
        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=len(dois), desc="Fetching metadata")
        
        for future in iterator:
            try:
                doi, metadata = future.result()
                results[doi] = metadata
            except Exception as e:
                doi = futures[future]
                print(f"Error with {doi}: {e}")
                results[doi] = None
    
    return results


if __name__ == "__main__":
    # Test with a known DOI
    test_doi = "10.1103/PhysRevC.108.034615"
    print(f"Testing with DOI: {test_doi}")
    result = fetch_metadata(test_doi)
    if result:
        print(f"Title: {result['title']}")
        print(f"Authors: {', '.join(result['authors'])}")
        print(f"Year: {result['year']}")
        print(f"Journal: {result['journal']}")
    else:
        print("No metadata found")