# backend.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import aiosqlite
import asyncio
import threading
from langchain_core.tools import tool, BaseTool
load_dotenv()
from typing import Dict, Any, Optional
import tempfile
import os
import sqlite3
# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"  # Or use "mixtral-8x7b-32768", etc.
)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    return _THREAD_RETRIEVERS.get(str(thread_id))

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        temp_path = f.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]
    finally:
        os.remove(temp_path)

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

# -------------------
# 2. Tools
# -------------------
# Tools

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve information from uploaded PDF for this chat thread.
    Always pass thread_id.
    """
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document uploaded for this thread.",
            "query": query
        }

    docs = retriever.invoke(query)

    return {
        "query": query,
        "context": [d.page_content for d in docs],
        "metadata": [d.metadata for d in docs],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }

search_tool = DuckDuckGoSearchRun(region="us-en")

# @tool
# def get_stock_price(symbol: str) -> dict:
#     """
#     Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
#     using Alpha Vantage with API key in the URL.
#     """
#     url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
#     r = requests.get(url)
#     return r.json()

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python",
            "args": ["./mcp_server.py"]
        }
        
    }
)
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []
    
mcp_tools = load_mcp_tools()

tools = [search_tool, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    thread_id = None
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")

    system_msg = HumanMessage(
        content=(
            "You are a helpful assistant.\n"
            "If the user asks about an uploaded document, "
            "call the `rag_tool` and include thread_id = "
            f"{thread_id}.\n"
            "Use web search or MCP tools when appropriate."
        )
    )

    messages = [system_msg, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------
async def _init_checkpointer():
    conn = await aiosqlite.connect("chatbot.db")

    # 🔥 PATCH for LangGraph bug
    if not hasattr(conn, "is_alive"):
        conn.is_alive = lambda: True

    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)


# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config["configurable"]["thread_id"])
#     return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})