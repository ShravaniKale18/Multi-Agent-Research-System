from langchain.tools import tool
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import requests

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_APIKEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Return titles, urls and snippets."""
    results=tavily.search(query=query, max_results=5)

    out=[]

    for result in results['results']:
        out.append(
            f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['content'][:300]}"
        )

    formatted_results = "\n----\n".join(out)

    return formatted_results

# print(web_search.invoke("Give me most recent news in india??"))

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

