"""
PDF text extraction module.
Uses PyMuPDF (fitz) for fast and accurate text extraction.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def extract_text(pdf_path: str | Path) -> Optional[str]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return None
        
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page in doc:
            text = page.get_text()
            if text:
                text_parts.append(text)
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        
        # Basic cleanup
        # Remove excessive whitespace
        import re
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)
        
        return full_text.strip() if full_text.strip() else None
        
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None


def extract_text_batch(
    pdf_paths: list[str | Path],
    max_workers: int = 4,
    progress: bool = True
) -> dict[str, Optional[str]]:
    """
    Extract text from multiple PDFs in parallel.
    
    Args:
        pdf_paths: List of paths to PDF files
        max_workers: Number of parallel workers
        progress: Show progress bar
    
    Returns:
        Dictionary mapping path -> extracted text (or None)
    """
    results = {}
    
    # Note: Using ThreadPoolExecutor instead of ProcessPoolExecutor
    # because fitz releases the GIL during processing
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_text, path): str(path) for path in pdf_paths}
        
        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=len(pdf_paths), desc="Extracting PDF text")
        
        for future in iterator:
            path = futures[future]
            try:
                text = future.result()
                results[path] = text
            except Exception as e:
                print(f"Error with {path}: {e}")
                results[path] = None
    
    return results


def get_pdf_info(pdf_path: str | Path) -> Optional[dict]:
    """
    Get basic info about a PDF (page count, metadata).
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary with PDF info or None
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return None
        
        doc = fitz.open(pdf_path)
        
        info = {
            "page_count": len(doc),
            "title": doc.metadata.get("title"),
            "author": doc.metadata.get("author"),
            "subject": doc.metadata.get("subject"),
            "keywords": doc.metadata.get("keywords"),
            "creator": doc.metadata.get("creator"),
            "producer": doc.metadata.get("producer"),
        }
        
        doc.close()
        return info
        
    except Exception as e:
        print(f"Error getting PDF info from {pdf_path}: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Extracting text from: {pdf_path}")
        
        text = extract_text(pdf_path)
        if text:
            print(f"Extracted {len(text)} characters")
            print("\n--- First 1000 characters ---")
            print(text[:1000])
        else:
            print("No text extracted")
    else:
        print("Usage: python pdf_extract.py <pdf_path>")