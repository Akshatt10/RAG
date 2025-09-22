from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from document import AdvancedRAGPipeline, rag_retriever, llm  
app = FastAPI()

adv_rag = AdvancedRAGPipeline(rag_retriever, llm)

class RAGQuery(BaseModel):
    question: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.2
    stream: Optional[bool] = False
    summarize: Optional[bool] = False

@app.post("/query")
async def query_rag(payload: RAGQuery):
    result = adv_rag.query(
        question=payload.question,
        top_k=payload.top_k,
        min_score=payload.min_score,
        stream=payload.stream,
        summarize=payload.summarize
    )
    return {"summary": result["summary"]}
