#!/usr/bin/env python
# coding: utf-8

# ### Data Ingestion
# 

# In[2]:


from langchain_core.documents import Document


# In[9]:


from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load every .docx file inside ../data/word (recursively if needed)
loader = DirectoryLoader(
    "../data/word",                      # Folder containing Word files
    glob="**/*.docx",                    # Match all .docx files (subfolders too)
    loader_cls=UnstructuredWordDocumentLoader,
    show_progress=True,
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"{len(chunks)} chunks ready for embeddings")


# In[10]:


chunks


# ### Embeddings and vector store db

# In[11]:


import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path


# In[12]:


def load_word_documents(word_dir: str):
    loader = DirectoryLoader(
        word_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True,
    )
    docs = loader.load()

    # Add metadata
    for doc in docs:
        doc.metadata['file_type'] = 'word'

    print(f"Loaded {len(docs)} Word documents")
    return docs

docs = load_word_documents("../data/word")


# In[13]:


docs


# In[14]:


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nExample chunk: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs

chunks = split_documents(docs)


# In[16]:


import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity


# In[17]:


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        print(f"Model '{self.model_name}' loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embedding(self, texts: list):
        return self.model.encode(texts, show_progress_bar=True)

embedding_manager = EmbeddingManager()


# ### Vector Store

# In[18]:


from typing import List, Any
import numpy as np
import os
import uuid
import chromadb

class VectorStore:
    """Manages document embeddings in a chromaDB vector store."""

    def __init__(self, collection_name: str= 'word_documents', persist_directory: str = "../data/vector_store"):

        """Initialize the VectorStore with ChromaDB.
        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to persist the ChromaDB data.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            #Create persistent chromadb client'
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Get or create collection
            self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "Document embeddings collection"})
            print(f"ChromaDB collection '{self.collection_name}' initialized successfully.")
            print(f"Existing number of documents in the collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
            """
            Add documents and their embeddings to the vector store

            Args:
                documents: List of LangChain documents
                embeddings: Corresponding embeddings for the documents
            """
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")

            print(f"Adding {len(documents)} documents to vector store...")

            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            documents_text = []
            embeddings_list = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Generate unique ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)

                # Prepare metadata
                metadata = dict(doc.metadata)
                metadata['doc_index'] = i
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                # Document content
                documents_text.append(doc.page_content)

                # Embedding
                embeddings_list.append(embedding.tolist())

            # Add to collection
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents_text
                )
                print(f"Successfully added {len(documents)} documents to vector store")
                print(f"Total documents in collection: {self.collection.count()}")

            except Exception as e:
                print(f"Error adding documents to vector store: {e}")
                raise

vectorstore=VectorStore()
vectorstore


# In[19]:


def load_word_documents(word_dir: str):
    loader = DirectoryLoader(
        word_dir,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        show_progress=True,
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata['file_type'] = 'word'

    print(f"Loaded {len(docs)} Word documents")
    return docs

docs = load_word_documents("../data/word")

# --------------------------
# 2. Split into chunks
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks ready for embeddings")

# --------------------------
# 3. Generate embeddings
# --------------------------
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_manager.generate_embedding(texts)

# --------------------------
# 4. Store in VectorStore
# --------------------------
vectorstore.add_documents(chunks, embeddings)


# In[20]:


chunks 


# In[21]:


### Converting chunks to embeddings and adding to vector store

texts = [doc.page_content for doc in chunks]

## Generate embeddings for the chunks
embeddings = embedding_manager.generate_embedding(texts)

### Store in the vector store
vectorstore.add_documents(chunks, embeddings)


# ## Retriever pipeline from vectorStore

# In[22]:


class RAGRetriever:
    """Retrieval-Augmented Generation (RAG) retriever using ChromaDB."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the RAG retriever.

        Args:
            vector_store (VectorStore): The vector store instance to use for retrieval.
            embedding_manager (EmbeddingManager): The embedding manager instance to generate embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)



# In[23]:


rag_retriever


# In[30]:


results = rag_retriever.retrieve("What are the primary methods to prevent and treat low blood sugar in a newborn?")
results


# ### RAG with groq LLM
# 

# In[31]:


from langchain_groq import ChatGroq
import  os
from dotenv import load_dotenv
load_dotenv()


### Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")

llm= ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it", temperature=0.1, max_tokens=1024)


# In[32]:


from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class GroqLLM:
    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str =None):
        """
        Initialize Groq LLM

        Args:
            model_name: Groq model name (qwen2-72b-instruct, llama3-70b-8192, etc.)
            api_key: Groq API key (or set GROQ_API_KEY environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )

        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """
        Generate response using retrieved context

        Args:
            query: User question
            context: Retrieved document context
            max_length: Maximum response length

        Returns:
            Generated response string
        """

        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )

        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation without complex prompting

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Generated response
        """
        simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""

        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"


# In[33]:


# Initialize Groq LLM (you'll need to set GROQ_API_KEY environment variable)
try:
    groq_llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq LLM initialized successfully!")
except ValueError as e:
    print(f"Warning: {e}")
    print("Please set your GROQ_API_KEY environment variable to use the LLM.")
    groq_llm = None


# In[35]:


### get the context from the retriever and pass it to the LLM

rag_retriever.retrieve("What are the primary methods to prevent and treat low blood sugar in a newborn?")


# In[36]:


### Simple RAG pipeline with Groq LLM
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

### Initialize the Groq LLM (set your GROQ_API_KEY in environment)
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0.1,max_tokens=1024)

## 2. Simple RAG function: retrieve context + generate response
def rag_simple(query,retriever,llm,top_k=3):
    ## retriever the context
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    ## generate the answwer using GROQ LLM
    prompt=f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}

        Answer:"""

    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content


# In[37]:


answer=rag_simple("What are the primary methods to prevent and treat low blood sugar in a newborn?",rag_retriever,llm)
print(answer)


# In[39]:


# --- Enhanced RAG Pipeline Features ---
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}

    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])

    # Generate answer
    prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context, query=query)])

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output

# Example usage:
result = rag_advanced("What are the primary methods to prevent and treat low blood sugar in a newborn?", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
print("Answer:", result['answer'])
print("Sources:", result['sources'])
print("Confidence:", result['confidence'])
print("Context Preview:", result['context'][:300])


# In[57]:


# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import List, Dict, Any
import time

class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
result = adv_rag.query("What is the recommended dose of Ampicillin for newborn infection prevention?", top_k=3, min_score=0.1, stream=True, summarize=True)
print("\nFinal Answer:", result['answer'])
print("Summary:", result['summary'])
print("History:", result['history'][-1])


# In[51]:




# In[49]:



# In[52]:



# In[ ]:




