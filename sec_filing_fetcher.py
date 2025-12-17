"""
SEC Filing Fetcher and Processor

This module fetches SEC 10-K/10-Q filings from EDGAR and preprocesses them
for extraction with LangExtract.

Key features:
- Fetch filings directly from SEC EDGAR
- Extract relevant sections (Item 1, Item 7, financial statements)
- Clean HTML and convert to readable text
- Handle rate limiting and caching
"""

import os
import re
import time
import requests
from typing import Optional, Dict, List
from bs4 import BeautifulSoup
from pathlib import Path


class SECFilingFetcher:
    """
    Fetches and processes SEC filings from EDGAR.
    
    Usage:
        fetcher = SECFilingFetcher()
        filing_text = fetcher.get_filing("NVDA", "10-K", "2024")
    """
    
    BASE_URL = "https://www.sec.gov"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; Research/1.0; your-email@example.com)"
    }
    
    def __init__(self, cache_dir: str = "./sec_filings_cache"):
        """
        Initialize the fetcher with optional caching.
        
        Args:
            cache_dir: Directory to cache downloaded filings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get the CIK (Central Index Key) for a company ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., "NVDA", "MSFT")
            
        Returns:
            CIK string or None if not found
        """
        # Simplified ticker to CIK mapping (in production, use SEC's ticker lookup API)
        ticker_to_cik = {
            "NVDA": "0001045810",
            "MSFT": "0000789019",
            "AAPL": "0000320193",
            "GOOGL": "0001652044",
            "AMZN": "0001018724",
            "TSLA": "0001318605",
            "META": "0001326801",
        }
        return ticker_to_cik.get(ticker.upper())
    
    def search_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search for recent filings of a specific type.
        
        Args:
            cik: Company CIK number
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            count: Number of recent filings to return
            
        Returns:
            List of filing metadata dicts
        """
        # Format CIK (remove leading zeros for URL)
        cik_formatted = cik.lstrip("0")
        
        # Search URL
        search_url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            "action": "getcompany",
            "CIK": cik_formatted,
            "type": filing_type,
            "dateb": "",
            "owner": "exclude",
            "count": count,
            "search_text": ""
        }
        
        print(f"Searching for {filing_type} filings for CIK {cik}...")
        
        try:
            response = requests.get(
                search_url,
                params=params,
                headers=self.HEADERS,
                timeout=10
            )
            response.raise_for_status()
            
            # Parse the results page
            soup = BeautifulSoup(response.content, "lxml")
            filings = []
            
            # Find filing entries in the table
            table = soup.find("table", {"class": "tableFile2"})
            if not table:
                print("No filings table found")
                return []
            
            rows = table.find_all("tr")[1:]  # Skip header row
            
            for row in rows[:count]:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    filing_date = cols[3].text.strip()
                    doc_link = cols[1].find("a")
                    
                    if doc_link:
                        doc_url = self.BASE_URL + doc_link["href"]
                        filings.append({
                            "filing_type": filing_type,
                            "filing_date": filing_date,
                            "filing_url": doc_url
                        })
            
            print(f"Found {len(filings)} filings")
            return filings
            
        except Exception as e:
            print(f"Error searching filings: {e}")
            return []
    
    def get_filing_text(
        self,
        filing_url: str,
        sections: Optional[List[str]] = None
    ) -> str:
        """
        Fetch and extract text from a filing URL.
        
        Args:
            filing_url: URL to the filing documents page
            sections: Optional list of sections to extract (e.g., ["Item 1", "Item 7"])
            
        Returns:
            Cleaned text content
        """
        # Check cache first
        cache_key = filing_url.split("/")[-1]
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            print(f"Using cached filing: {cache_file}")
            return cache_file.read_text(encoding="utf-8")
        
        print(f"Fetching filing from: {filing_url}")
        
        try:
            # Get the filing page
            response = requests.get(filing_url, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            
            # Find the main document link (usually .htm or .html)
            soup = BeautifulSoup(response.content, "lxml")
            table = soup.find("table", {"class": "tableFile"})
            
            if not table:
                print("Could not find document table")
                return ""
            
            # Find the main document (usually labeled as "10-K" or "10-Q")
            doc_link = None
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    doc_type = cols[3].text.strip()
                    if doc_type in ["10-K", "10-Q", "10-K/A", "10-Q/A"]:
                        link = cols[2].find("a")
                        if link:
                            doc_link = self.BASE_URL + link["href"]
                            break
            
            if not doc_link:
                print("Could not find main document link")
                return ""
            
            # Fetch the actual document
            print(f"Downloading document: {doc_link}")
            time.sleep(0.1)  # Rate limiting
            
            doc_response = requests.get(doc_link, headers=self.HEADERS, timeout=30)
            doc_response.raise_for_status()
            
            # Parse HTML and extract text
            doc_soup = BeautifulSoup(doc_response.content, "lxml")
            
            # Extract specific sections if requested
            if sections:
                text = self._extract_sections(doc_soup, sections)
            else:
                text = self._extract_full_text(doc_soup)
            
            # Cache the result
            cache_file.write_text(text, encoding="utf-8")
            print(f"Cached filing to: {cache_file}")
            
            return text
            
        except Exception as e:
            print(f"Error fetching filing: {e}")
            return ""
    
    def _extract_full_text(self, soup: BeautifulSoup) -> str:
        """Extract all text from the document."""
        # Remove script and style elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        
        return text.strip()
    
    def _extract_sections(self, soup: BeautifulSoup, sections: List[str]) -> str:
        """
        Extract specific sections from the filing.
        
        This is a simplified implementation. In production, you'd need more
        sophisticated parsing to handle various filing formats.
        """
        full_text = self._extract_full_text(soup)
        
        # For now, just return the full text
        # In production, implement section extraction based on Item numbers
        return full_text[:50000]  # Limit to first 50k chars for testing
    
    def get_filing(
        self,
        ticker: str,
        filing_type: str = "10-K",
        year: Optional[str] = None,
        sections: Optional[List[str]] = None
    ) -> str:
        """
        Convenience method to fetch a filing by ticker and type.
        
        Args:
            ticker: Stock ticker (e.g., "NVDA")
            filing_type: Type of filing (10-K, 10-Q)
            year: Optional year filter (e.g., "2024")
            sections: Optional sections to extract
            
        Returns:
            Filing text content
        """
        # Get CIK
        cik = self.get_company_cik(ticker)
        if not cik:
            print(f"CIK not found for ticker: {ticker}")
            return ""
        
        # Search for filings
        filings = self.search_filings(cik, filing_type, count=10)
        
        if not filings:
            print(f"No {filing_type} filings found for {ticker}")
            return ""
        
        # Filter by year if specified
        if year:
            filings = [f for f in filings if year in f["filing_date"]]
        
        if not filings:
            print(f"No {filing_type} filings found for {ticker} in {year}")
            return ""
        
        # Get the most recent filing
        filing = filings[0]
        print(f"\nFetching {ticker} {filing_type} from {filing['filing_date']}")
        
        return self.get_filing_text(filing["filing_url"], sections=sections)


def main():
    """
    Example usage of SECFilingFetcher.
    """
    fetcher = SECFilingFetcher()
    
    # Fetch NVIDIA's most recent 10-K
    print("="*60)
    print("Fetching NVIDIA 10-K Filing")
    print("="*60 + "\n")
    
    filing_text = fetcher.get_filing(
        ticker="NVDA",
        filing_type="10-K",
        sections=None  # Get full filing (will be truncated)
    )
    
    if filing_text:
        print(f"\n✓ Retrieved filing: {len(filing_text):,} characters")
        print("\nFirst 500 characters:")
        print("-" * 60)
        print(filing_text[:500])
        print("-" * 60)
        
        # Save to file
        output_file = "nvidia_10k_sample.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(filing_text)
        print(f"\n✓ Saved to: {output_file}")
    else:
        print("\n✗ Failed to retrieve filing")


if __name__ == "__main__":
    main()
