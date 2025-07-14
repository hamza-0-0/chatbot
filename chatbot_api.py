import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Initialize embedding and load existing vector DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="db", embedding_function=embedding)

# Set up retriever with top-k
k = 1
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

# Create FastAPI app
app = FastAPI()

# Request model with optional category
class QueryRequest(BaseModel):
    question: str
    category: Optional[str] = None

@app.post("/ask")
def ask_question(req: QueryRequest):
    kwargs = {}
    if req.category:
        kwargs["filter"] = {"category": req.category}

    results = retriever.get_relevant_documents(req.question, **kwargs)

    if results:
        answers = [
            doc.page_content.split("A:")[1].strip() if "A:" in doc.page_content else doc.page_content
            for doc in results
        ]
        return {"answers": answers}
    else:
        return {"answers": ["Sorry, I couldn't find an answer."]}
