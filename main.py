from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# =========================
# 🔑 CREATE GEMINI CLIENT
# =========================
client = genai.Client(api_key="AIzaSyA-KOVK79fgmtSDjV3DIc1CfIyLQjf1CoQ")


# =========================
# 🔹 EMBEDDING FUNCTION
# =========================
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


# =========================
# 🔹 CUSTOM EMBEDDING CLASS
# =========================
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [get_embedding(t) for t in texts]

    def embed_query(self, text):
        return get_embedding(text)


# =========================
# 🔹 LOAD PDF
# =========================
loader = PyPDFLoader("data.pdf")
docs = loader.load()


# =========================
# 🔹 SPLIT INTO CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)


# =========================
# 🔹 CREATE VECTOR DB
# =========================
embeddings = GeminiEmbeddings()
db = FAISS.from_documents(chunks, embeddings)


# =========================
# 🔹 USER QUERY
# =========================
query = input("\nAsk a question: ")


# =========================
# 🔹 SEARCH
# =========================
results = db.similarity_search(query, k=3)

context = "\n\n".join([r.page_content for r in results])


# =========================
# 🔹 PROMPT
# =========================
prompt = f"""
You are analyzing documents.

Rules:
- Only answer using the context
- If not found, say "Not found in document"
- Do NOT assume anything
- Say "mentioned in document"

Context:
{context}

Question:
{query}
"""


# =========================
# 🔹 GENERATE ANSWER
# =========================
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


# =========================
# 🔹 PRINT
# =========================
print("\nAnswer:\n")
print(response.text)


# =========================
# 🔹 SOURCES
# =========================
print("\nSources:\n")

for i, r in enumerate(results):
    print(f"Source {i+1} (Page {r.metadata.get('page', 'N/A')}):")
    print(r.page_content[:300])
    print("-----\n")
