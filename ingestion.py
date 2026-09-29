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



async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )


    # Create batches
    batches = [
        documents[i : i+batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )



    # To process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True



    # Process batches concurrently (Hitting the rate limit so I will process sequentially)
    # tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)


    # Process batches sequentially to avoid session closing or rate limits
    results = []
    for i, batch in enumerate(batches):
        success = await add_batch(batch, i + 1)
        results.append(success)
        await asyncio.sleep(4)


    # Count successful batches
    successful = sum(1 for result in results if result is True)


    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )



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
        "max_depth": 3,         # How deep crawler goes from the main URL
        "extract_depth": "advanced",            # To try to get tables and dynamic objects
        "instructions": "content on ai agents"          # Instruction in natural language for the crawler AI
    })




    # Convert Tavily crawl results to LangChain Document objects
    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )


    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )


    # Clear existing vectors to avoid duplicates
    vectorstore.delete(delete_all=True)


    # Process document asynchronously
    await index_documents_async(splitted_docs, batch_size=5)


    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())