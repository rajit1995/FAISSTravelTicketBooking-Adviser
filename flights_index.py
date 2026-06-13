"""
flights_index.py
Builds a FAISS vector index from flights.json.

Changes from original:
- tqdm progress bar so the user can see real-time progress
- Price formatted back to ₹-string in the document text for LLM readability
  (the raw int is kept in metadata for potential filtering)
- Graceful error if flights.json is missing
- Configurable BATCH_SIZE and INDEX_PATH
"""

import json
import os
import sys

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    print("⚠️  tqdm not installed — install it for progress bars: pip install tqdm")

# ── Config ────────────────────────────────────────────────────────────────────
FLIGHTS_FILE = "flights.json"
INDEX_PATH   = "flights_index"
BATCH_SIZE   = 50_000
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load data
if not os.path.exists(FLIGHTS_FILE):
    print(f"❌ {FLIGHTS_FILE} not found. Run flight_data_generator.py first.")
    sys.exit(1)

with open(FLIGHTS_FILE, "r", encoding="utf-8") as f:
    flights = json.load(f)

print(f"📂 Loaded {len(flights):,} flight records from {FLIGHTS_FILE}")

# 2. Build document strings and metadata
docs      = []
metadatas = []
for fl in flights:
    price_str = f"₹{fl['price']}" if isinstance(fl["price"], int) else fl["price"]
    docs.append(
        f"Flight option: {fl['airline']} from {fl['origin']} to {fl['destination']} "
        f"on {fl['departure']} costing {price_str}"
    )
    metadatas.append(fl)

# 3. Embeddings
print(f"🔧 Loading embedding model: {EMBED_MODEL}")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# 4. Build FAISS index in batches
vector_db  = None
total      = len(docs)
batches    = range(0, total, BATCH_SIZE)

iterator = tqdm(batches, desc="Building index", unit="batch") if USE_TQDM else batches

for i in iterator:
    batch_docs = docs[i : i + BATCH_SIZE]
    batch_meta = metadatas[i : i + BATCH_SIZE]

    batch_db = FAISS.from_texts(batch_docs, embeddings, metadatas=batch_meta)

    if vector_db is None:
        vector_db = batch_db
    else:
        vector_db.merge_from(batch_db)

    if not USE_TQDM:
        done = min(i + BATCH_SIZE, total)
        print(f"  ✅ Processed {done:,} / {total:,} records …")

# 5. Persist
vector_db.save_local(INDEX_PATH)
print(f"\n📦 Index saved to '{INDEX_PATH}/' ({total:,} records in total)")
