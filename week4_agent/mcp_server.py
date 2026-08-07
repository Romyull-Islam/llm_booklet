# week4_agent/mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("knowledge-base")

@mcp.tool
def search_docs(query: str, top_k: int = 5) -> str:
    """Search the internal knowledge base and return relevant passages."""
    # call your Week 3 retriever here
    return f"results for: {query}"

@mcp.resource("docs://{doc_id}")
def get_doc(doc_id: str) -> str:
    """Fetch a full document by ID."""
    return f"contents of {doc_id}"

if __name__ == "__main__":
    mcp.run()
