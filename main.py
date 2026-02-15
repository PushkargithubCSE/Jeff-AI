from google import genai
import os 
import time 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# =========================
# 🔑 CREATE GEMINI CLIENT
# =========================
client = genai.Client(api_key="AIzaSyDc3TSPjJBJLkX0REZ5SHhGn_bJ0fSAOa4")   


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
# 🔹 SEARCH PERSON FUNCTION
# =========================
def search_person(name, chunks):
    results = []

    for c in chunks:
        if name.lower() in c.page_content.lower():
            results.append(c)

    return results


# =========================
# 🔹 CUSTOM EMBEDDING CLASS
# =========================
# 🔴 UPDATED (BATCHING ADDED)
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []

        batch_size = 20   # 🔴 safe for free tier

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            print(f"Embedding batch {i} → {i + len(batch)}")

            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch
                )

                for emb in response.embeddings:
                    embeddings.append(emb.values)

            except Exception as e:
                print("⚠️ Error, retrying...", e)
                time.sleep(5)
                continue

            time.sleep(1)   # 🔴 IMPORTANT (avoid rate limit)

        return embeddings

    def embed_query(self, text):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return response.embeddings[0].values


# =========================
# 🔹 LOAD PDF
# =========================
all_docs = []

folder_path = "Data"

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".pdf"):
            file_path = os.path.join(root, filename)

            print(f"Loading {filename}...")

            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                for d in docs:
                    d.metadata["source"] = filename

                all_docs.extend(docs)

            except Exception as e:
                print(f"❌ Skipping {filename}: {e}")

print(f"\nTotal documents loaded: {len(all_docs)}")


# =========================
# 🔹 SPLIT INTO CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)


# =========================
# 🔹 CREATE VECTOR DB
# =========================
# 🔴 ADDED PRINT (progress visibility)
print("\nCreating embeddings & vector DB... (this may take time)\n")

embeddings = GeminiEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

print("\n✅ Vector DB created\n")


# =========================
# 🔹 USER QUERY
# =========================
mode = input("\nType 'ask' or 'person': ").strip().lower()


# =========================
# PERSON MODE
# =========================
if mode == "person":
    name = input("Enter name: ")

    results = search_person(name, chunks)

    print(f"\nFound {len(results)} mentions for '{name}':\n")

    if len(results) == 0:
        print("No mentions found.")
    else:
        for i, r in enumerate(results[:20]):
            print(f"{i+1}. File: {r.metadata.get('source')}")
            print(f"   Page: {r.metadata.get('page')}")
            print(r.page_content[:300])
            print("-----\n")


# =========================
# ASK MODE
# =========================
elif mode == "ask":
    query = input("Ask a question: ")

    results = db.similarity_search(query, k=5)

    context = "\n\n".join([r.page_content for r in results])

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAnswer:\n")
    print(response.text)

    print("\nSources:\n")

    for i, r in enumerate(results):
        print(f"{i+1}. File: {r.metadata.get('source')}, Page: {r.metadata.get('page')}")
        print(r.page_content[:200])
        print("-----\n")


else:
    print("Invalid option")
