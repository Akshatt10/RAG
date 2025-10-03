from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from document import retriever

app = FastAPI()

# ---------------- CORS ----------------
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # React default (if using create-react-app)
    "*"  # or allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],   # allow custom headers
)
# --------------------------------------

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
        summarize=payload.summarize,
    )
    return result

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    result = retriever.upload_document(file)
    return result

@app.get("/")
async def root():
    return {"message": "RAG API is running! Put your documents in the 'data' folder."}
