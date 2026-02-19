from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from rag import build_rag

app = FastAPI()

templates = Jinja2Templates(directory="templates")

rag_chain = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": None}
    )


@app.post("/build", response_class=HTMLResponse)
def build(request: Request, video_id: str = Form(...)):
    global rag_chain

    rag_chain = build_rag(video_id)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "message": "RAG Built Successfully ✅"
        }
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    if rag_chain is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "answer": "Build RAG first!"}
        )

    answer = rag_chain.invoke(question)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "answer": answer}
    )
