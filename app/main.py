from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import BaseModel, Field

from .graph import ask


app = FastAPI(

    title="Agentic AI RAG Chatbot",

    description=(
        "Grounded RAG chatbot based "
        "only on the Agentic AI eBook."
    ),

    version="1.0.0"
)


class ChatRequest(BaseModel):

    question: str = Field(

        ...,

        min_length=2,

        max_length=1000
    )


@app.get("/")
def root():

    return {

        "message":
        "Agentic AI RAG Chatbot API",

        "docs":
        "/docs"
    }


@app.get("/health")
def health():

    return {

        "status":
        "healthy"
    }


@app.post("/chat")
def chat(
    request: ChatRequest
):

    try:

        result = ask(
            request.question
        )

        return result

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )