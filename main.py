from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag_backend import load_and_index_documents, generate_rag_response


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Load retriever once at startup
retriever_instance = load_and_index_documents()
if retriever_instance is None:
    raise RuntimeError("No documents indexed. Add PDFs to docs folder.")


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
    global chat_history

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)