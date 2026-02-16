from fastapi import FastAPI
from pydantic import BaseModel
from main import generate_answer_safe, search_person, find_connections, load_and_process_pdfs
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from google import genai
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load DB once
DB_PATH = "faiss_index"

client = genai.Client(api_key="AIzaSyBlq7sprKpm7wUW37w-a_EDe6nUzMe7cbk")

# reuse embeddings
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        pass
    def embed_query(self, text):
        pass

embeddings = DummyEmbeddings()

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# load chunks for person/connection
chunks = load_and_process_pdfs()


class QueryRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(req: QueryRequest):
    results = db.similarity_search(req.query, k=5)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
    Context:
    {context}

    Question:
    {req.query}
    """

    answer = generate_answer_safe(prompt)

    return {
        "answer": answer,
        "sources": [
            {
                "file": r.metadata.get("source"),
                "page": r.metadata.get("page")
            } for r in results
        ]
    }


@app.get("/person/{name}")
def person(name: str):
    results = search_person(name, chunks)

    return {
        "count": len(results),
        "results": [
            {
                "file": r.metadata.get("source"),
                "page": r.metadata.get("page"),
                "text": r.page_content[:300]
            } for r in results[:20]
        ]
    }


@app.get("/connections/{name}")
def connections(name: str):
    conn = find_connections(name, chunks)

    return {
        "connections": conn
    }
