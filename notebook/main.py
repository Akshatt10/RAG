from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from document import retriever 

app = FastAPI()

class RAGQuery(BaseModel):
    question: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.3
    summarize: Optional[bool] = True
@app.post("/query")
async def query_rag(payload: RAGQuery):
    result = retriever.query(
        question=payload.question,
        top_k=payload.top_k,
        min_score=payload.min_score,
        summarize=payload.summarize
    )
    return result

@app.get("/")
async def root():
    return {"message": "RAG API is running! Put your documents in the 'data' folder."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)