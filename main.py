from google import genai
import os
import time
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# =========================
# 🔑 CREATE GEMINI CLIENT
# =========================
# Ensure your API key is correct
client = genai.Client(api_key="AIzaSyBlq7sprKpm7wUW37w-a_EDe6nUzMe7cbk")


# =========================
# 🔹 CUSTOM EMBEDDING CLASS (ROBUST VERSION)
# =========================
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        batch_size = 20  # Safe batch size for free tier

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # print(f"Embedding batch {i} → {i + len(batch)}") # Optional: comment out to reduce noise

            success = False
            while not success:
                try:
                    response = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch
                    )
                    for emb in response.embeddings:
                        embeddings.append(emb.values)
                    success = True
                    time.sleep(1) # Small pause to be safe
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print("⚠️ Embedding Quota hit. Waiting 30s...")
                        time.sleep(30)
                    else:
                        print(f"❌ Error: {e}")
                        time.sleep(5)
        return embeddings

    def embed_query(self, text):
        while True:
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text
                )
                return response.embeddings[0].values
            except Exception as e:
                if "429" in str(e):
                    print("⚠️ Quota hit on query. Waiting 10s...")
                    time.sleep(10)
                else:
                    print(f"Error embedding query: {e}")
                    return []


# =========================
# 🔹 LOAD & PROCESS DATA
# =========================
def load_and_process_pdfs():
    all_docs = []
    folder_path = "Data"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder '{folder_path}'. Please put PDFs inside.")
        return []

    print("\nLoading PDF files...")
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = filename
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"❌ Skipping {filename}: {e}")

    print(f"Total documents loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=100
    )
    return splitter.split_documents(all_docs)


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
# 🔹 GENERATE ANSWER (WITH RETRY)
# =========================
def generate_answer_safe(prompt):
    """Retries the generation if quota is hit."""
    while True:
        try:
            # 🔴 CHANGED MODEL TO 'gemini-1.5-flash' (More stable free tier)
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("⚠️ Generation Quota hit. Waiting 30s...")
                time.sleep(30)
            else:
                return f"Error generating answer: {e}"


# =========================
# 🔹 MAIN EXECUTION
# =========================
DB_PATH = "faiss_index"
embeddings = GeminiEmbeddings()

# Ask Mode first to determine if we need to load DB or Raw Text
mode = input("\nType 'ask' or 'person': ").strip().lower()

if mode == "person":
    chunks = load_and_process_pdfs()
    if not chunks:
        print("No data found.")
        exit()

    name = input("Enter name: ")
    results = search_person(name, chunks)
    print(f"\nFound {len(results)} mentions for '{name}':\n")
    
    for i, r in enumerate(results[:20]):
        print(f"{i+1}. File: {r.metadata.get('source')} (Page: {r.metadata.get('page')})")
        print(r.page_content[:300].replace('\n', ' '))
        print("-----\n")

elif mode == "ask":
    # Check for existing DB
    if os.path.exists(DB_PATH):
        print("\n✅ Loading existing Vector DB from disk...")
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("\nNo local DB found. Processing PDFs...")
        chunks = load_and_process_pdfs()
        if not chunks:
            print("No data found.")
            exit()
            
        print("Creating embeddings & vector DB... (this may take time)")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_PATH)
        print(f"✅ Vector DB saved to '{DB_PATH}'\n")

    # Chat Loop
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == 'exit':
            break

        results = db.similarity_search(query, k=5)
        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""
        You are an expert investigator analyzing legal documents.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        INSTRUCTIONS:
        - Answer strictly based on the provided Context.
        - If the answer is not in the context, state "Information not found in the provided documents."
        """

        print("\nThinking...")
        answer = generate_answer_safe(prompt)
        print("\n=== ANSWER ===\n")
        print(answer)
        
        print("\n=== SOURCES ===")
        seen_sources = set()
        for r in results:
            source_info = f"{r.metadata.get('source')} (Page {r.metadata.get('page')})"
            if source_info not in seen_sources:
                print(f"• {source_info}")
                seen_sources.add(source_info)

else:
    print("Invalid option")