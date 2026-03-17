from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag_backend import load_and_index_documents, generate_rag_response


app = FastAPI()

templates = Jinja2Templates(directory="templates")

retriever_instance = None

chat_history = [
    {"bot": "Hi, I'm your AI assistant."}
]


class QueryRequest(BaseModel):
    query: str


# Root (for Render health check)
@app.get("/")
async def root():
    return {"status": "running"}


#  UI route
@app.get("/chat", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history
        }
    )


# RAG endpoint
@app.post("/ask")
async def ask(query_request: QueryRequest):
    global chat_history, retriever_instance

    query = query_request.query.strip()

    if not query:
        return JSONResponse({
            "user": "",
            "bot": "Please enter a question."
        })

    # Lazy load 
    if retriever_instance is None:
        print("Loading RAG pipeline...")
        retriever_instance = load_and_index_documents()

    response = generate_rag_response(query, retriever_instance)

    chat_history.append({
        "user": query,
        "bot": response
    })

    return JSONResponse({
        "user": query, 
        "bot": response
    })