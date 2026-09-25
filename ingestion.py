import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma         #In case we want to use Chroma instead of Pinecone
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter          #Split the document
from langchain_core.documents import Document           #Text Doc with associated metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings         #Google Gemini embedding
from langchain_pinecone import PineconeVectorStore          #Pinecone multidimensional vector store
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap          #Tavily for different tools including crawling and search

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)          #From the logger file


# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

### Embedding ###

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# In case of using open source embedding like Chroma:
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"])


### Crawling ###

# Tavily extract and map will be needed in case of manual crawl.
# tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

site_to_crawl = "https://python.langchain.com/"





async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        f"🗺️  TavilyCrawl: Starting to crawl the documentation from {site_to_crawl}",
        Colors.PURPLE
    )

    # Crawl the documentation site

    res = tavily_crawl.invoke({
        "url": f"{site_to_crawl}",
        "max_depth": 5,         # How deep crawler goes from the main URL
        "extract_depth": "advanced",            # To try to get tables and dynamic objects
        "instructions": "content on ai agents"          # Instruction in natural language for the crawler AI
    })
 
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["result"]
    ]

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )



if __name__ == "__main__":
    asyncio.run(main())