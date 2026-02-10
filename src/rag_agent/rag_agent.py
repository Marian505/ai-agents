import base64
from io import BytesIO
import asyncio
import os
from typing import Any
import bs4
from dotenv import load_dotenv
from pypdf import PdfReader
import tempfile

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_agent
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_postgres import PGVector

load_dotenv()

POSTGRES_STORE = os.getenv("POSTGRES_STORE")

model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
if POSTGRES_STORE:
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="rag_docs",
        connection=POSTGRES_STORE,
        use_jsonb=True,
        async_mode=True,
    )
else:
    vector_store = InMemoryVectorStore(embedding=embeddings)

def _split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

async def load_web(web_paths: list[str]) -> int:
    """Function to load PDF."""
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(web_paths=web_paths, bs_kwargs={"parse_only": bs4_strainer})
    docs = loader.load()
    chunks = await asyncio.to_thread(_split_documents, docs)
    vector_store.collection_name = "web_docs_"
    doc_ids = await vector_store.aadd_documents(documents=chunks)
    return len(doc_ids)

async def load_pdf_bytes(pdf_bytes: bytes) -> int:
    """Function to load PDF."""
    tmp_path = await asyncio.to_thread(_create_temp_pdf, pdf_bytes)
    try:
        loader = PyMuPDFLoader(tmp_path)
        docs = await loader.aload()
        chunks = await asyncio.to_thread(_split_documents, docs)
        doc_ids = await vector_store.aadd_documents(documents=chunks)
        return len(doc_ids)
    finally:
        await asyncio.to_thread(os.unlink, tmp_path)

def _create_temp_pdf(pdf_bytes: bytes) -> str:
    """Sync helper - runs safely in thread."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        return tmp.name

async def load_pdfs(pdf_paths: list[str]) -> int:
    """Function to load PDF."""
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = await loader.aload()
        chunks = await asyncio.to_thread(_split_documents, docs) 
        # set metadata per document if needed
        doc_ids = await vector_store.aadd_documents(documents=chunks)
    return len(doc_ids)

def _base64_to_pdf(file):
    base64_data = file.split("base64,", 1)[1]
    pdf_bytes = base64.b64decode(base64_data)
    pdf_file = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    return reader.pages

# example of loading data via middleweare, does not work well
@before_agent
async def load_pdfs_middleweare(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Function to lad PDF as agent middleweare."""
    last_message = state['messages'][-1].content
    if isinstance(last_message, str):
        return None
    if isinstance(last_message, list):
        files = [item.get("file", {}).get("file_data") 
                    for item in last_message 
                        if item.get("type") == "file" and item.get("file", {}).get("file_data") is not None]
        
        docs = []
        for file in files:
            pages = await asyncio.to_thread(_base64_to_pdf, file)         
            for page_num, page in enumerate(pages):
                text = page.extract_text()
                docs.append(Document(page_content=text, metadata={"page": page_num, "source": "pdf_file"}))

        chunks = await asyncio.to_thread(_split_documents, docs) 
        _doc_ids = await vector_store.aadd_documents(documents=chunks)

        return_message = next((item.get("text") for item in last_message if item.get("type") == "text"), None)
        
        # TODO: maybe add ai message, pdf is loaded, update retriver to retrive only source related data by filter
        return {"messages": state['messages'][:-1] + [HumanMessage(content=return_message)]}

@tool
async def retrieve_context(query: str):
    """Retrieve information related to the query."""
    retrieved_docs = await vector_store.asimilarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Id: {doc.id}\nSource: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized

system_prompt = """
You have access to loaded PDF documents via the retrieve_context tool.
Always use retrieve_context first when answering questions about uploaded documents.
Reply only informations from retrieved documents and do not use any other knowledge you have.
If you don't find the answer in retrieved documents, reply that you don't know, but never use any other knowledge you have.
"""

agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt=system_prompt,
)
