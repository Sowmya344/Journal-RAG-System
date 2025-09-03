from sentence_transformers import SentenceTransformer
from supabase import create_client
from tqdm import tqdm
import time

# --- Supabase Setup ---
supabase_url = "https://nobnxgtmpspfvhhxtlvw.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vYm54Z3RtcHNwZnZoaHh0bHZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0NDIxMjgsImV4cCI6MjA2NTAxODEyOH0.BqdlpaONkK0UkF1BY-i0dqu1W9czgj3LCasEttCneDo"
supabase = create_client(supabase_url, supabase_key)

# --- Model Setup ---
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# --- Constants ---
BATCH_SIZE = 16
MAX_WORDS = 512

def truncate_text(text):
    return " ".join(text.split()[:MAX_WORDS])

# --- Fetch all articles ---
response = supabase.table("articles").select("id, full_text").execute()
articles = response.data

# --- Prepare data ---
entries = [(a["id"], truncate_text(a["full_text"]))
           for a in articles if a.get("id") and a.get("full_text")]

print(f"🧠 Generating embeddings for {len(entries)} articles...")

# --- Generate and upload embeddings in batches ---
for i in tqdm(range(0, len(entries), BATCH_SIZE)):
    batch = entries[i:i + BATCH_SIZE]
    ids = [entry[0] for entry in batch]
    texts = [entry[1] for entry in batch]

    try:
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=BATCH_SIZE)

        for article_id, embedding in zip(ids, embeddings):
            supabase.table("articles").update({"embedding": embedding.tolist()}).eq("id", article_id).execute()
            time.sleep(0.2)  # throttle to avoid rate limits

    except Exception as e:
        print(f"\n❌ Error in batch {i // BATCH_SIZE + 1}: {e}")

print("\n✅ All embeddings updated successfully.")