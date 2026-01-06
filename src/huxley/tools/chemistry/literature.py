"""
Scientific literature search tools for Huxley.

Provides access to:
- arXiv API (free, no auth required)
- PubMed API (free, optional API key for higher rate limits)
"""

import asyncio
import httpx
import xml.etree.ElementTree as ET
from typing import Any
from datetime import datetime
import html
import re


async def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search arXiv for scientific papers.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        sort_by: Sort order (relevance, lastUpdatedDate, submittedDate)
        categories: Optional list of arXiv categories to filter
                   (e.g., ["q-bio.BM", "physics.bio-ph", "cs.AI"])
        
    Returns:
        Dict with search results
    """
    base_url = "https://export.arxiv.org/api/query"
    
    # Build search query
    search_query = query.replace(" ", "+AND+")
    
    # Add category filters if specified
    if categories:
        cat_filter = "+OR+".join([f"cat:{cat}" for cat in categories])
        search_query = f"({search_query})+AND+({cat_filter})"
    
    # Sort options
    sort_map = {
        "relevance": "relevance",
        "lastUpdatedDate": "lastUpdatedDate", 
        "submittedDate": "submittedDate",
    }
    sort_order = sort_map.get(sort_by, "relevance")
    
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": sort_order,
        "sortOrder": "descending",
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Define namespaces
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            
            # Extract total results
            total_results = root.find("opensearch:totalResults", 
                {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"})
            total = int(total_results.text) if total_results is not None else 0
            
            papers = []
            for entry in root.findall("atom:entry", ns):
                # Extract paper info
                paper_id = entry.find("atom:id", ns)
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                updated = entry.find("atom:updated", ns)
                
                # Get authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                
                # Get categories
                categories_found = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories_found.append(term)
                
                # Get PDF link
                pdf_link = None
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href")
                
                # Extract arXiv ID from URL
                arxiv_id = paper_id.text.split("/abs/")[-1] if paper_id is not None else ""
                
                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title.text.strip().replace("\n", " ") if title is not None else "",
                    "authors": authors[:5],  # Limit authors
                    "author_count": len(authors),
                    "abstract": summary.text.strip().replace("\n", " ")[:500] if summary is not None else "",
                    "published": published.text[:10] if published is not None else "",
                    "updated": updated.text[:10] if updated is not None else "",
                    "categories": categories_found[:3],
                    "pdf_url": pdf_link,
                    "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                })
            
            return {
                "success": True,
                "source": "arXiv",
                "query": query,
                "total_results": total,
                "returned": len(papers),
                "papers": papers,
            }
            
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {str(e)}"}
    except ET.ParseError as e:
        return {"error": f"XML parse error: {str(e)}"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


async def search_pubmed(
    query: str,
    max_results: int = 10,
    sort: str = "relevance",
) -> dict[str, Any]:
    """
    Search PubMed for biomedical literature.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        sort: Sort order (relevance, pub_date)
        
    Returns:
        Dict with search results
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    # First, search for IDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 100),
        "sort": sort,
        "retmode": "json",
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Search for IDs
            search_response = await client.get(f"{base_url}/esearch.fcgi", params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            result = search_data.get("esearchresult", {})
            id_list = result.get("idlist", [])
            total = int(result.get("count", 0))
            
            if not id_list:
                return {
                    "success": True,
                    "source": "PubMed",
                    "query": query,
                    "total_results": 0,
                    "returned": 0,
                    "papers": [],
                }
            
            # Fetch details for IDs
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
            
            fetch_response = await client.get(f"{base_url}/efetch.fcgi", params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(fetch_response.text)
            
            papers = []
            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                if medline is None:
                    continue
                
                pmid = medline.find(".//PMID")
                article_data = medline.find(".//Article")
                
                if article_data is None:
                    continue
                
                # Title
                title_elem = article_data.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Abstract
                abstract_elem = article_data.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else ""
                
                # Authors
                authors = []
                for author in article_data.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)
                
                # Date
                pub_date = article_data.find(".//PubDate")
                year = pub_date.find("Year").text if pub_date is not None and pub_date.find("Year") is not None else ""
                
                # Journal
                journal = article_data.find(".//Journal/Title")
                journal_name = journal.text if journal is not None else ""
                
                # DOI
                doi = None
                for id_elem in article.findall(".//ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text
                
                papers.append({
                    "pmid": pmid.text if pmid is not None else "",
                    "title": title,
                    "authors": authors[:5],
                    "author_count": len(authors),
                    "abstract": abstract[:500] if abstract else "",
                    "year": year,
                    "journal": journal_name,
                    "doi": doi,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid.text}/" if pmid is not None else "",
                })
            
            return {
                "success": True,
                "source": "PubMed",
                "query": query,
                "total_results": total,
                "returned": len(papers),
                "papers": papers,
            }
            
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


async def get_paper_details(
    paper_id: str,
    source: str = "arxiv",
) -> dict[str, Any]:
    """
    Get detailed information about a specific paper.
    
    Args:
        paper_id: Paper ID (arXiv ID or PMID)
        source: Source database (arxiv or pubmed)
        
    Returns:
        Dict with paper details
    """
    if source.lower() == "arxiv":
        # Query arXiv for specific paper
        base_url = "http://export.arxiv.org/api/query"
        params = {"id_list": paper_id}
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(base_url, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                entry = root.find("atom:entry", ns)
                if entry is None:
                    return {"error": f"Paper {paper_id} not found"}
                
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                
                return {
                    "success": True,
                    "source": "arXiv",
                    "arxiv_id": paper_id,
                    "title": title.text.strip() if title is not None else "",
                    "abstract": summary.text.strip() if summary is not None else "",
                    "authors": authors,
                    "published": published.text if published is not None else "",
                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                    "abs_url": f"https://arxiv.org/abs/{paper_id}",
                }
                
        except Exception as e:
            return {"error": f"Failed to fetch paper: {str(e)}"}
    
    elif source.lower() == "pubmed":
        # Query PubMed
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": paper_id,
            "retmode": "xml",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(base_url, params=params)
                response.raise_for_status()
                
                root = ET.fromstring(response.text)
                article = root.find(".//PubmedArticle")
                
                if article is None:
                    return {"error": f"Paper {paper_id} not found"}
                
                medline = article.find(".//MedlineCitation")
                article_data = medline.find(".//Article")
                
                title_elem = article_data.find(".//ArticleTitle")
                abstract_elem = article_data.find(".//Abstract/AbstractText")
                
                authors = []
                for author in article_data.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)
                
                return {
                    "success": True,
                    "source": "PubMed",
                    "pmid": paper_id,
                    "title": title_elem.text if title_elem is not None else "",
                    "abstract": abstract_elem.text if abstract_elem is not None else "",
                    "authors": authors,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                }
                
        except Exception as e:
            return {"error": f"Failed to fetch paper: {str(e)}"}
    
    else:
        return {"error": f"Unknown source: {source}. Use 'arxiv' or 'pubmed'"}
