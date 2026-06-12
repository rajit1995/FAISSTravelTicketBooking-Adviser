import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

with open("flights.json", "r") as f:
    flights = json.load(f)

docs = [
    f"Airline: {f['airline']}, Origin: {f['origin']}, Destination: {f['destination']}, Price: {f['price']}, Departure: {f['departure']}"
    for f in flights
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_texts(docs, embeddings)
vector_db.save_local("flights_index")
print("📂 flights_index built successfully!")
