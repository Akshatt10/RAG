# import os
# import uuid
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Any
# import chromadb
# from sentence_transformers import SentenceTransformer
# from langchain_core.documents import Document
# from langchain_community.document_loaders import (
#     DirectoryLoader, UnstructuredWordDocumentLoader, PyPDFLoader, 
#     TextLoader, UnstructuredMarkdownLoader, CSVLoader,
#     JSONLoader, UnstructuredHTMLLoader
# )
# from fastapi import UploadFile
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage
# from dotenv import load_dotenv

# class SimpleRAGSystem:
#     """Simple RAG system that auto-loads from data folder."""
    
#     SUPPORTED_EXTENSIONS = {
#         '.pdf': PyPDFLoader,
#         '.docx': UnstructuredWordDocumentLoader,
#         '.doc': UnstructuredWordDocumentLoader,
#         '.txt': TextLoader,
#         '.md': UnstructuredMarkdownLoader,
#         '.csv': CSVLoader,
#         '.json': JSONLoader,
#         '.html': UnstructuredHTMLLoader,
#         '.htm': UnstructuredHTMLLoader,
#     }
    
#     def __init__(self, data_folder: str = "../data"):
#         self.data_folder = Path(data_folder)
#         self.data_folder.mkdir(exist_ok=True)
        
#         # Initialize embedding model
#         self.embedding_model = SentenceTransformer("BAAI/bge-large-en")
        
#         # Initialize vector store
#         self.client = chromadb.PersistentClient(path="vector_store")
#         self.collection = self.client.get_or_create_collection(name="documents")
        
#         # Initialize LLM
#         load_dotenv()
#         self.llm = ChatGoogleGenerativeAI(
#             google_api_key=os.getenv("GOOGLE_API_KEY"),
#             model="gemini-2.5-pro",
#             temperature=0.1
#         )
        
#         # Text splitter
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=250, 
#             chunk_overlap=50
#         )
        
#         # Auto-load documents on startup
#         self._load_all_documents()
    
#     def _load_single_file(self, file_path: Path) -> List[Document]:
#         """Load a single file."""
#         extension = file_path.suffix.lower()
        
#         if extension not in self.SUPPORTED_EXTENSIONS:
#             print(f"Skipping unsupported file: {file_path}")
#             return []
            
#         loader_class = self.SUPPORTED_EXTENSIONS[extension]
        
#         try:
#             if extension == '.json':
#                 loader = loader_class(str(file_path), jq_schema='.')
#             else:
#                 loader = loader_class(str(file_path))
                
#             docs = loader.load()
            
#             # Add metadata
#             for doc in docs:
#                 doc.metadata.update({
#                     'source_file': str(file_path),
#                     'file_name': file_path.name,
#                     'file_type': extension[1:]
#                 })
                
#             print(f"Loaded {len(docs)} chunks from {file_path.name}")
#             return docs
            
#         except Exception as e:
#             print(f"Error loading {file_path}: {e}")
#             return []
    
#     def _load_all_documents(self):
#         """Load all documents from data folder."""
#         print(f"Loading documents from {self.data_folder}")
        
#         # Check if already loaded
#         if self.collection.count() > 0:
#             print(f"Found {self.collection.count()} existing documents in vector store")
#             return
        
#         all_docs = []
        
#         # Find all supported files
#         for extension in self.SUPPORTED_EXTENSIONS.keys():
#             files = list(self.data_folder.rglob(f"*{extension}"))
#             for file_path in files:
#                 docs = self._load_single_file(file_path)
#                 all_docs.extend(docs)
        
#         if not all_docs:
#             print("No documents found in data folder")
#             return
            
#         # Split into chunks
#         chunks = self.splitter.split_documents(all_docs)
#         print(f"Split into {len(chunks)} chunks")
        
#         # Generate embeddings and store
#         texts = [chunk.page_content for chunk in chunks]
#         embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
#         # Prepare for ChromaDB
#         ids = []
#         metadatas = []
#         documents = []
#         embeddings_list = []
        
#         for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#             ids.append(f"doc_{i}")
#             metadatas.append(chunk.metadata)
#             documents.append(chunk.page_content)
#             embeddings_list.append(embedding.tolist())
        
#         # Add to vector store
#         self.collection.add(
#             ids=ids,
#             embeddings=embeddings_list,
#             metadatas=metadatas,
#             documents=documents
#         )
        
#         print(f"Added {len(chunks)} documents to vector store")
        
#     def upload_document(self, file: UploadFile) -> Dict[str, Any]:
#         """Upload a single document and add it to the vector store."""
#         try:
#             # Save file temporarily in data folder
#             file_path = self.data_folder / file.filename
#             with open(file_path, "wb") as f:
#                 f.write(file.file.read())
            
#             # Load the document
#             docs = self._load_single_file(file_path)
#             if not docs:
#                 return {"status": "error", "message": "Unsupported or empty file."}
            
#             # Split into chunks
#             chunks = self.splitter.split_documents(docs)
#             texts = [chunk.page_content for chunk in chunks]
#             embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
#             # Prepare vectors
#             ids = [str(uuid.uuid4()) for _ in chunks]
#             metadatas = [chunk.metadata for chunk in chunks]
#             documents = texts
#             embeddings_list = [emb.tolist() for emb in embeddings]
            
#             # Add to vector store
#             self.collection.add(
#                 ids=ids,
#                 embeddings=embeddings_list,
#                 metadatas=metadatas,
#                 documents=documents
#             )
            
#             return {
#                 "status": "success",
#                 "file": file.filename,
#                 "chunks_added": len(chunks)
#             }
        
#         except Exception as e:
#             return {"status": "error", "message": str(e)}
    
#     def query(self, question: str, top_k: int = 5, min_score: float = 0.3, summarize: bool = True) -> Dict[str, Any]:
#         """Query the knowledge base."""
        
#         # Generate query embedding
#         query_embedding = self.embedding_model.encode([question])[0]
        
#         # Search vector store
#         results = self.collection.query(
#             query_embeddings=[query_embedding.tolist()],
#             n_results=top_k
#         )
        
#         if not results['documents'] or not results['documents'][0]:
#             return {
#                 'answer': 'No relevant information found in your documents.',
#                 'sources': [],
#                 'confidence': 0.0,
#                 'summary': 'No information available to summarize.'
#             }
        
#         # Filter by score
#         documents = results['documents'][0]
#         metadatas = results['metadatas'][0]
#         distances = results['distances'][0]
        
#         filtered_docs = []
#         for doc, metadata, distance in zip(documents, metadatas, distances):
#             similarity_score = 1 - distance
#             if similarity_score >= min_score:
#                 filtered_docs.append({
#                     'content': doc,
#                     'metadata': metadata,
#                     'score': similarity_score
#                 })
        
#         if not filtered_docs:
#             return {
#                 'answer': 'No relevant information found above the confidence threshold.',
#                 'sources': [],
#                 'confidence': 0.0,
#                 'summary': 'No information available to summarize.'
#             }
        
#         # Prepare context
#         context = "\n\n".join([doc['content'] for doc in filtered_docs])
        
#         # Generate detailed answer
#         prompt = f"""Answer the question based on the following context from the documents.
#         Provide a comprehensive and well-structured response.
        
#         Context:
#         {context}
        
#         Question: {question}
        
#         Answer:"""
        
#         try:
#             response = self.llm.invoke([HumanMessage(content=prompt)])
#             answer = response.content
#         except Exception as e:
#             answer = f"Error generating response: {str(e)}"
        
#         # Generate enhanced summary with bullet points
#         summary = None
#         if summarize and answer:
#             summary_prompt = f"""You are creating a summary for a user who needs quick, actionable insights. Analyze the answer and provide ONLY the most essential information.

#             Rules for the summary:
#             1. Maximum 3 bullet points
#             2. Focus on WHAT the user needs to DO or KNOW first
#             3. Highlight prerequisites, warnings, or critical steps
#             4. Use action-oriented language (verbs like "Ensure", "Install", "Configure")
#             5. Avoid repeating obvious information
#             6. If there are multiple steps, group them logically
            
#             Answer to summarize:
#             {answer}
            
#             Essential Summary (3 bullet points maximum, focus on action and prerequisites):"""
            
#             try:
#                 summary_resp = self.llm.invoke([HumanMessage(content=summary_prompt)])
#                 summary = summary_resp.content
#             except Exception as e:
#                 summary = "Unable to generate summary."
#         # Prepare sources with citations
#         sources = [{
#             'file': doc['metadata'].get('file_name', 'unknown'),
#             'type': doc['metadata'].get('file_type', 'unknown'),
#             'score': doc['score'],
#             'preview': doc['content'][:120] + '...'
#         } for doc in filtered_docs]
        
#         # Add citations to answer
#         citations = [f"[{i+1}] {src['file']}" for i, src in enumerate(sources)]
#         answer_with_citations = answer + "\n\nSources:\n" + "\n".join(citations) if citations else answer
        
#         confidence = max([doc['score'] for doc in filtered_docs])
        
#         return {
#             'answer': answer_with_citations,
#             'sources': sources,
#             'confidence': confidence,
#             'summary': summary
#         }


# # Initialize the system
# rag_system = SimpleRAGSystem()

# # For compatibility with your existing code
# class RAGRetriever:
#     def __init__(self, rag_system):
#         self.rag_system = rag_system
    
#     def query(self, question: str, top_k: int = 5, min_score: float = 0.3, **kwargs):
#         return self.rag_system.query(question, top_k, min_score)
    
#     def upload_document(self, file: UploadFile) -> Dict[str, Any]:
#         return self.rag_system.upload_document(file)

# # Create the retriever instance for your main.py
# retriever = RAGRetriever(rag_system)
# llm = rag_system.llm  # For compatibility





import os
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader, PyPDFLoader,
    TextLoader, UnstructuredMarkdownLoader, CSVLoader,
    JSONLoader, UnstructuredHTMLLoader
)
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv


# --- Developer-friendly response wrapper ---
class RAGResponse:
    def __init__(self, answer, summary, sources, confidence):
        self.answer = answer
        self.summary = summary
        self.sources = sources
        self.confidence = confidence

    def __repr__(self):
        sources_str = "\n".join(
            [f"- {s['file']} (score={s['score']:.2f})" for s in self.sources]
        ) or "None"
        return (
            f"\n🔎 Answer:\n{self.answer}\n\n"
            f"📌 Summary:\n{self.summary}\n\n"
            f"📂 Sources:\n{sources_str}\n\n"
            f"✅ Confidence: {self.confidence:.2f}\n"
        )


class SimpleRAGSystem:
    """Simple but developer-friendly RAG system."""

    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
        '.json': JSONLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
    }

    def __init__(self, data_folder: str = "../data"):
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)

        # Embedding model
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en")

        # Vector store
        self.client = chromadb.PersistentClient(path="vector_store")
        self.collection = self.client.get_or_create_collection(name="documents")

        # LLM
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-2.5-pro",
            temperature=0.1
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50
        )

        # Auto-load docs on startup
        self._load_all_documents()

    # --- File loading ---
    def _load_single_file(self, file_path: Path) -> List[Document]:
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file: {file_path}")
            return []

        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        try:
            if extension == '.json':
                loader = loader_class(str(file_path), jq_schema='.')
            else:
                loader = loader_class(str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata.update({
                    'source_file': str(file_path),
                    'file_name': file_path.name,
                    'file_type': extension[1:]
                })
            return docs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def _load_all_documents(self):
        if self.collection.count() > 0:
            print(f"Vector store already has {self.collection.count()} documents.")
            return

        all_docs = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in self.data_folder.rglob(f"*{ext}"):
                all_docs.extend(self._load_single_file(file_path))

        if not all_docs:
            print("No documents found.")
            return

        chunks = self.splitter.split_documents(all_docs)
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        self.collection.add(
            ids=[f"doc_{i}" for i in range(len(chunks))],
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=[c.metadata for c in chunks],
            documents=texts
        )
        print(f"Indexed {len(chunks)} chunks into vector store.")

    # --- Helpers for developers ---
    def list_files(self):
        data = self.collection.get()
        return list({m.get("file_name", "unknown") for m in data.get("metadatas", [])})

    def clear_index(self):
        self.collection.delete(ids=self.collection.get()["ids"])
        return "Vector store cleared."

    # --- Upload ---
    def upload_document(self, file: UploadFile) -> Dict[str, Any]:
        try:
            file_path = self.data_folder / file.filename
            with open(file_path, "wb") as f:
                f.write(file.file.read())

            docs = self._load_single_file(file_path)
            if not docs:
                return {"status": "error", "message": "Unsupported or empty file."}

            chunks = self.splitter.split_documents(docs)
            texts = [c.page_content for c in chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

            ids = [str(uuid.uuid4()) for _ in chunks]
            self.collection.add(
                ids=ids,
                embeddings=[e.tolist() for e in embeddings],
                metadatas=[c.metadata for c in chunks],
                documents=texts
            )
            return {"status": "success", "file": file.filename, "chunks_added": len(chunks)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # --- Query ---
    def query(self, question: str, top_k: int = 5, min_score: float = 0.3, mode="qa", summarize: bool = True) -> Any:
        query_emb = self.embedding_model.encode([question])[0]
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        if not results['documents'] or not results['documents'][0]:
            return RAGResponse("No results found.", "None", [], 0.0)

        documents, metadatas, distances = (
            results['documents'][0], results['metadatas'][0], results['distances'][0]
        )

        filtered = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1 - dist
            if score >= min_score:
                filtered.append({"content": doc, "metadata": meta, "score": score})

        if not filtered:
            return RAGResponse("No relevant info above threshold.", "None", [], 0.0)

        context = "\n\n".join([d["content"] for d in filtered])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        except Exception as e:
            answer = f"Error generating response: {e}"

        # Summary
        summary = None
        if mode != "raw":
            try:
                summary_prompt = f"Summarize in 3 bullet points:\n{answer}"
                summary_resp = self.llm.invoke([HumanMessage(content=summary_prompt)])
                summary = summary_resp.content
            except Exception:
                summary = "Summary unavailable."

        sources = [{
            "file": d["metadata"].get("file_name", "unknown"),
            "type": d["metadata"].get("file_type", "unknown"),
            "score": d["score"],
            "preview": d["content"][:100] + "..."
        } for d in filtered]

        confidence = max([d["score"] for d in filtered])

        if mode == "raw":
            return {
                "answer": answer,
                "summary": summary,
                "sources": sources,
                "confidence": confidence
            }
        elif mode == "summary":
            return summary
        else:
            return RAGResponse(answer, summary, sources, confidence)


# --- Retriever wrapper (for external use) ---
class RAGRetriever:
    def __init__(self, rag_system):
        self.rag_system = rag_system

    def query(self, question: str, top_k: int = 5, min_score: float = 0.3,summarize: bool =True, **kwargs):
        return self.rag_system.query(question, top_k, min_score,summarize, **kwargs)

    def upload_document(self, file: UploadFile) -> Dict[str, Any]:
        return self.rag_system.upload_document(file)


# Initialize
rag_system = SimpleRAGSystem()
retriever = RAGRetriever(rag_system)
llm = rag_system.llm
