import json
import gradio as gr
from datetime import datetime
import torch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
import os

# -------------------------------
# Prompt Template
# -------------------------------
booking_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. "
               "Your job is to evaluate flight options and recommend the best one."),
    ("human", "Book a {trip_type} flight from {origin} to {destination} on {date}. "
              "If return, also book return flight on {return_date}.")
])

# -------------------------------
# Initialize Ollama with CUDA
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {device}")
if device == "cuda":
    print("🎯 GPU:", torch.cuda.get_device_name(0))

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.4,
    top_p=0.9,
    num_ctx=4096,
    device=device
)

# -------------------------------
# Load FAISS Index
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

index_path = r"C:\Users\LOQ\Documents\Rajit_Roy\Zensar\Trainings\AI Training Zensar\Flight Tickets\flights_index"

if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
    vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ flights_index loaded successfully!")
else:
    print("⚠️ flights_index not found. Please build index first.")
    vector_db = None

# -------------------------------
# RAG Search Function
# -------------------------------
def rag_search(query, k=20):
    if vector_db is None:
        return []
    results = vector_db.similarity_search(query, k=k)
    return [r.page_content for r in results]

# -------------------------------
# Booking Workflow with RAG + LLM Analysis
# -------------------------------
def travel_booking_workflow(origin, destination, day, month, year, trip_type,
                            return_day=None, return_month=None, return_year=None):
    date = f"{year}-{int(month):02d}-{int(day):02d}"
    return_date = None
    if trip_type == "return" and return_day and return_month and return_year:
        return_date = f"{return_year}-{int(return_month):02d}-{int(return_day):02d}"

    query = f"Flights from {origin} to {destination} on {date} under ₹8000"
    flights = rag_search(query, k=20)

    if not flights:
        return f"❌ No flights found from {origin} to {destination} near {date}."

    # Let the LLM decide cheapest vs expensive
    flights_text = "\n".join(flights)
    analysis_prompt = (
        f"Here are the available flights:\n{flights_text}\n\n"
        f"Please identify the cheapest and most expensive flights, "
        f"compare them, and recommend the best option. "
        f"Then provide a booking confirmation message."
    )
    response = llm.invoke(analysis_prompt).content

    result = f"🔎 Analysis & Booking:\n{response}"

    if trip_type == "return" and return_date:
        query_return = f"Flights from {destination} to {origin} on {return_date} under ₹7000"
        return_flights = rag_search(query_return, k=20)
        if not return_flights:
            result += f"\n\n❌ No return flights found from {destination} to {origin} near {return_date}."
        else:
            return_flights_text = "\n".join(return_flights)
            return_analysis_prompt = (
                f"Here are the available return flights:\n{return_flights_text}\n\n"
                f"Please identify the cheapest and most expensive return flights, "
                f"compare them, and recommend the best option. "
                f"Then provide a booking confirmation message."
            )
            return_response = llm.invoke(return_analysis_prompt).content
            result += f"\n\n🔎 Return Analysis & Booking:\n{return_response}"

    return result

# -------------------------------
# Gradio Interface
# -------------------------------
cities = ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bangalore",
          "Hyderabad", "Pune", "Jaipur", "Ahmedabad", "Lucknow"]

days = list(range(1, 32))
months = list(range(1, 13))
years = [2026, 2027, 2028]

with gr.Blocks() as demo:
    gr.Markdown("## ✈️ Travel Ticket Booking Assistant (RAG + FAISS + LLM Analysis)")

    with gr.Row():
        origin = gr.Dropdown(choices=cities, label="Origin City")
        destination = gr.Dropdown(choices=cities, label="Destination City")

    with gr.Row():
        day = gr.Dropdown(choices=days, label="Day")
        month = gr.Dropdown(choices=months, label="Month")
        year = gr.Dropdown(choices=years, label="Year")

    trip_type = gr.Radio(choices=["one way", "return"], label="Trip Type")

    with gr.Row():
        return_day = gr.Dropdown(choices=days, label="Return Day", visible=False)
        return_month = gr.Dropdown(choices=months, label="Return Month", visible=False)
        return_year = gr.Dropdown(choices=years, label="Return Year", visible=False)

    def toggle_return_fields(trip_type):
        visible = trip_type == "return"
        return {return_day: gr.update(visible=visible),
                return_month: gr.update(visible=visible),
                return_year: gr.update(visible=visible)}

    trip_type.change(toggle_return_fields, inputs=trip_type,
                     outputs=[return_day, return_month, return_year])

    submit_btn = gr.Button("Book Ticket")
    output = gr.Textbox(label="Result", lines=20)

    submit_btn.click(travel_booking_workflow,
                     inputs=[origin, destination, day, month, year, trip_type,
                             return_day, return_month, return_year],
                     outputs=output)

# -------------------------------
# Launch App
# -------------------------------
if __name__ == "__main__":
    demo.launch()
