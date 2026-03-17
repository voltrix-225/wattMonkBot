from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag_backend import load_and_index_documents, generate_rag_response


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Load retriever once at startup
retriever_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever_instance
    print("Loading documents and building retriever...")
    retriever_instance = load_and_index_documents()
    yield
    print("Shutting down application")

app = FastAPI(lifespan=lifespan)



chat_history = [
    {"bot": "Hi, I'm your AI assistant."}
]


class QueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history
        }
    )


@app.post("/ask")
async def ask(query_request: QueryRequest):
    global chat_history, retriever_instance

    # Lazy load here
    if retriever_instance is None:
        print("Loading RAG pipeline...")
        retriever_instance = load_and_index_documents()

    query = query_request.query.strip()

    if not query:
        return JSONResponse({
            "user": "",
            "bot": "Please enter a question."
        })

    response = generate_rag_response(query, retriever_instance)

    chat_history.append({
        "user": query,
        "bot": response
    })

    return JSONResponse({
        "user": query,
        "bot": response
    })

@app.get("/")
async def root():
    return {"message": "App is running"}