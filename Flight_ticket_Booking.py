"""
Flight_ticket_Booking.py
RAG-powered flight booking assistant — full interactive workflow:
  1. Search  → retrieve flights via FAISS
  2. LLM     → formats options, sorts by departure ASC, returns structured JSON
  3. UI      → customer picks a flight via Radio buttons
  4. LLM     → generates booking summary + payment invoice
  5. Payment → simulated (random txn-id, UPI/card mock)
  6. Email   → sends HTML confirmation to rajit1995@gmail.com via SMTP/Gmail
"""

import json
import os
import random
import re
import smtplib
import string
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import gradio as gr
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# ── Config ────────────────────────────────────────────────────────────────────
CUSTOMER_EMAIL = "rajit1995@gmail.com"
SENDER_EMAIL   = "rajitglasgow@gmail.com"
# For Gmail: generate an App Password at https://myaccount.google.com/apppasswords
# and set env var GMAIL_APP_PASSWORD, or paste it directly below.
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "cynsqputhihitffq")

CITIES      = ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bangalore",
               "Hyderabad", "Pune", "Jaipur", "Ahmedabad", "Lucknow"]
DAYS        = list(range(1, 32))
MONTHS      = list(range(1, 13))
YEARS       = [2026, 2027, 2028]
INDEX_PATH  = "flights_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_OPTIONS = 10   # max flights shown to customer
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3.2:latest", temperature=0.2, top_p=0.9, num_ctx=4096)

# ── Embeddings & Retriever ────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL, model_kwargs={"device": device}
)

retriever = None
if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")) and \
   os.path.exists(os.path.join(INDEX_PATH, "index.pkl")):
    vector_db = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 40})
    print("flights_index loaded successfully!")
else:
    print("flights_index not found. Run flights_index.py first.")


# ── Prompts ───────────────────────────────────────────────────────────────────

SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a flight search engine. Given raw flight data, extract and return "
     "ONLY a valid JSON array (no markdown, no explanation) of flight objects sorted "
     "by departure datetime ASCENDING. Each object must have exactly these keys: "
     "index (1-based int), airline, origin, destination, departure, price_inr (int). "
     f"Return at most {MAX_OPTIONS} flights. Example:\n"
     # NOTE: all {{ and }} are escaped so LangChain does not treat them as prompt variables
     '[{{"index":1,"airline":"IndiGo","origin":"Kolkata","destination":"Delhi",'
     '"departure":"2026-06-20 06:00","price_inr":4800}}]'),
    ("human",
     "Route: {origin} to {destination}, around {date} (plus or minus 3 days).\n"
     "Raw flight data:\n{context}\n\n"
     "Return the JSON array now."),
])

CONFIRM_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional airline booking assistant. "
     "Write a warm, detailed booking confirmation message for the customer. "
     "Include: flight details, PNR, seat class (Economy), baggage allowance (15 kg), "
     "check-in opens 2 hours before departure, and a thank-you note. "
     "Do NOT mention payment — that is handled separately."),
    ("human",
     "Customer email: {email}\n"
     "Chosen flight:\n{flight_json}\n"
     "Trip type: {trip_type}\n"
     "Return date: {return_date}\n"
     "PNR: {pnr}\n\n"
     "Write the booking confirmation now."),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date(day, month, year, label="Date"):
    try:
        dt = datetime(int(year), int(month), int(day))
        return dt.strftime("%Y-%m-%d"), None
    except ValueError:
        return None, f"Invalid {label}: {year}-{int(month):02d}-{int(day):02d} doesn't exist."


def build_context(origin, destination, date):
    if retriever is None:
        return ""
    query = f"flights from {origin} to {destination} on {date}"
    docs  = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)


def llm_search_flights(origin, destination, date):
    context = build_context(origin, destination, date)
    if not context:
        return None, "No flight data available (index missing)."

    chain = SEARCH_PROMPT | llm | StrOutputParser()
    raw   = chain.invoke({"origin": origin, "destination": destination,
                          "date": date, "context": context})

    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        flights = json.loads(raw)
        if not isinstance(flights, list) or len(flights) == 0:
            return None, "LLM returned no flights. Try different dates."
        # Re-assign index after sort (LLM may mis-number)
        for i, f in enumerate(flights):
            f["index"] = i + 1
        return flights, None
    except json.JSONDecodeError:
        return None, f"LLM response parse error. Raw output:\n{raw[:500]}"


def generate_pnr():
    return "PNR" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def simulate_payment(price_inr):
    methods = ["UPI (rajit1995@okaxis)", "Visa card ending 4242",
               "Mastercard ending 1234", "Net Banking - SBI"]
    txn_id  = "TXN" + uuid.uuid4().hex[:10].upper()
    return {
        "txn_id":    txn_id,
        "method":    random.choice(methods),
        "amount":    price_inr,
        "status":    "SUCCESS",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def llm_confirmation_message(flight, trip_type, return_date, pnr):
    chain = CONFIRM_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "email":       CUSTOMER_EMAIL,
        "flight_json": json.dumps(flight, indent=2),
        "trip_type":   trip_type,
        "return_date": return_date,
        "pnr":         pnr,
    })


def build_email_html(flight, payment, confirmation_text, pnr):
    price_fmt = f"Rs.{payment['amount']:,}"
    conf_escaped = confirmation_text.replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:620px;margin:auto;color:#333">
    <div style="background:#1a73e8;padding:24px;border-radius:8px 8px 0 0">
      <h1 style="color:white;margin:0">Booking Confirmed!</h1>
      <p style="color:#cce4ff;margin:6px 0 0 0">SkyAssist Travel &mdash; Your Journey Awaits</p>
    </div>
    <div style="border:1px solid #ddd;border-top:none;padding:24px;border-radius:0 0 8px 8px">
      <h2 style="color:#1a73e8">Flight Details</h2>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <tr><td style="padding:8px;background:#f8f9fa;font-weight:bold;width:160px">PNR</td>
            <td style="padding:8px">{pnr}</td></tr>
        <tr><td style="padding:8px;font-weight:bold">Airline</td>
            <td style="padding:8px">{flight['airline']}</td></tr>
        <tr><td style="padding:8px;background:#f8f9fa;font-weight:bold">Route</td>
            <td style="padding:8px">{flight['origin']} &rarr; {flight['destination']}</td></tr>
        <tr><td style="padding:8px;font-weight:bold">Departure</td>
            <td style="padding:8px">{flight['departure']}</td></tr>
        <tr><td style="padding:8px;background:#f8f9fa;font-weight:bold">Price</td>
            <td style="padding:8px;color:#2e7d32;font-weight:bold">{price_fmt}</td></tr>
      </table>

      <h2 style="color:#1a73e8;margin-top:28px">Payment Receipt</h2>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <tr><td style="padding:8px;background:#f8f9fa;font-weight:bold;width:160px">Transaction ID</td>
            <td style="padding:8px">{payment['txn_id']}</td></tr>
        <tr><td style="padding:8px;font-weight:bold">Payment Method</td>
            <td style="padding:8px">{payment['method']}</td></tr>
        <tr><td style="padding:8px;background:#f8f9fa;font-weight:bold">Amount Paid</td>
            <td style="padding:8px;font-weight:bold">{price_fmt}</td></tr>
        <tr><td style="padding:8px;font-weight:bold">Status</td>
            <td style="padding:8px;color:#2e7d32;font-weight:bold">SUCCESS</td></tr>
        <tr><td style="padding:8px;background:#f8f9fa;font-weight:bold">Timestamp</td>
            <td style="padding:8px">{payment['timestamp']}</td></tr>
      </table>

      <div style="background:#f0f7ff;border-left:4px solid #1a73e8;
                  padding:16px;margin-top:28px;border-radius:4px">
        <pre style="white-space:pre-wrap;font-family:Arial,sans-serif;
                    font-size:13px;margin:0;line-height:1.6">{conf_escaped}</pre>
      </div>

      <p style="color:#aaa;font-size:11px;margin-top:24px;text-align:center">
        This is a simulated booking for demonstration purposes only.<br>
        SkyAssist Travel Assistant &copy; 2026
      </p>
    </div>
    </body></html>
    """


def send_email(subject, html_body):
    if GMAIL_APP_PASSWORD == "YOUR_APP_PASSWORD_HERE":
        return False, ("Email skipped — set GMAIL_APP_PASSWORD env var or paste your "
                       "Gmail App Password in the script to enable sending.")
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = CUSTOMER_EMAIL
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, [CUSTOMER_EMAIL], msg.as_string())

        return True, "Confirmation email sent to " + CUSTOMER_EMAIL
    except Exception as e:
        return False, f"Email error: {e}"


# ── Gradio step functions ─────────────────────────────────────────────────────

def do_search(origin, destination, day, month, year, trip_type,
              return_day, return_month, return_year):
    no_radio  = gr.update(choices=[], visible=False)
    hide_sel  = gr.update(visible=False)
    hide_conf = gr.update(visible=False)

    if not origin or not destination:
        return ("Please select both origin and destination city.",
                no_radio, [], {}, hide_sel, hide_conf)
    if origin == destination:
        return ("Origin and destination cannot be the same city.",
                no_radio, [], {}, hide_sel, hide_conf)

    date_str, err = parse_date(day, month, year, "departure date")
    if err:
        return err, no_radio, [], {}, hide_sel, hide_conf

    return_date_str = "N/A"
    if trip_type == "return":
        if not (return_day and return_month and return_year):
            return ("Please select a return date for a return trip.",
                    no_radio, [], {}, hide_sel, hide_conf)
        return_date_str, err = parse_date(return_day, return_month, return_year, "return date")
        if err:
            return err, no_radio, [], {}, hide_sel, hide_conf
        if return_date_str <= date_str:
            return ("Return date must be after the departure date.",
                    no_radio, [], {}, hide_sel, hide_conf)

    flights, err = llm_search_flights(origin, destination, date_str)
    if err:
        return err, no_radio, [], {}, hide_sel, hide_conf

    # Build radio labels — sorted by departure (LLM already sorted, but re-sort defensively)
    flights.sort(key=lambda f: f["departure"])
    for i, f in enumerate(flights):
        f["index"] = i + 1

    choices = [
        f"{f['index']:>2}.  {f['airline']:<12}  |  "
        f"{f['origin']} -> {f['destination']}  |  "
        f"Rs.{f['price_inr']:,}  |  Dep: {f['departure']}"
        for f in flights
    ]

    meta = {"trip_type": trip_type, "return_date": return_date_str, "date": date_str}

    return (
        f"Found {len(flights)} flights from {origin} to {destination} on {date_str}. "
        "Please pick one below:",
        gr.update(choices=choices, value=None, visible=True),
        flights,
        meta,
        gr.update(visible=True),
        gr.update(visible=False),
    )


def do_confirm(selected_label, flights_state, meta_state):
    if not selected_label:
        return "Please select a flight first.", gr.update(visible=False)
    if not flights_state:
        return "No flight data found. Please search again.", gr.update(visible=False)

    idx    = int(selected_label.strip().split(".")[0]) - 1
    flight = flights_state[idx]
    pnr    = generate_pnr()

    confirm_text = llm_confirmation_message(
        flight, meta_state["trip_type"], meta_state["return_date"], pnr
    )
    payment = simulate_payment(flight["price_inr"])

    lines = [
        "=" * 58,
        "  BOOKING CONFIRMATION",
        "=" * 58,
        confirm_text,
        "",
        "=" * 58,
        "  PAYMENT RECEIPT  (SIMULATED)",
        "=" * 58,
        f"  Transaction ID  : {payment['txn_id']}",
        f"  Payment Method  : {payment['method']}",
        f"  Amount Paid     : Rs.{payment['amount']:,}",
        f"  Status          : {payment['status']}",
        f"  Timestamp       : {payment['timestamp']}",
        "=" * 58,
    ]

    html    = build_email_html(flight, payment, confirm_text, pnr)
    subject = (f"Booking Confirmed - {pnr} | {flight['airline']} "
               f"{flight['origin']} to {flight['destination']}")
    ok, em  = send_email(subject, html)
    lines.append(f"\nEMAIL: {em}")

    return "\n".join(lines), gr.update(visible=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="SkyAssist Flight Booking", theme=gr.themes.Soft()) as demo:

    gr.Markdown("## SkyAssist — Flight Booking Assistant")
    gr.Markdown(
        "Powered by **LangChain · Ollama Llama 3.2 · FAISS**  \n"
        "**Step 1** — Search  →  **Step 2** — Pick your flight  →  **Step 3** — Confirm & Pay"
    )

    # ── Step 1: Search form ───────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("### Step 1 — Journey Details")
        with gr.Row():
            origin      = gr.Dropdown(choices=CITIES, label="Origin City")
            destination = gr.Dropdown(choices=CITIES, label="Destination City")

        gr.Markdown("**Departure Date**")
        with gr.Row():
            day   = gr.Dropdown(choices=DAYS,   label="Day",   value=20)
            month = gr.Dropdown(choices=MONTHS, label="Month", value=6)
            year  = gr.Dropdown(choices=YEARS,  label="Year",  value=2026)

        trip_type = gr.Radio(choices=["one way", "return"],
                             label="Trip Type", value="one way")

        with gr.Group(visible=False) as return_group:
            gr.Markdown("**Return Date**")
            with gr.Row():
                return_day   = gr.Dropdown(choices=DAYS,   label="Return Day")
                return_month = gr.Dropdown(choices=MONTHS, label="Return Month")
                return_year  = gr.Dropdown(choices=YEARS,  label="Return Year")

        trip_type.change(
            lambda c: gr.update(visible=(c == "return")),
            inputs=[trip_type], outputs=[return_group]
        )

        search_btn = gr.Button("Search Flights", variant="primary")

    status_box = gr.Textbox(label="Status", lines=2, interactive=False)

    # ── Step 2: Flight selection ──────────────────────────────────────────────
    with gr.Group(visible=False) as selection_group:
        gr.Markdown("### Step 2 — Select Your Flight  *(sorted by departure, earliest first)*")
        flight_radio = gr.Radio(choices=[], label="Available Flights", interactive=True)
        confirm_btn  = gr.Button("Confirm & Pay", variant="primary")

    # ── Step 3: Confirmation output ───────────────────────────────────────────
    with gr.Group(visible=False) as confirm_group:
        gr.Markdown("### Step 3 — Booking Confirmation & Payment")
        confirm_output = gr.Textbox(label="Booking Details", lines=30, interactive=False)

    # ── State ──────────────────────────────────────────────────────────────────
    flights_state = gr.State([])
    meta_state    = gr.State({})

    # ── Wiring ─────────────────────────────────────────────────────────────────
    search_btn.click(
        do_search,
        inputs=[origin, destination, day, month, year, trip_type,
                return_day, return_month, return_year],
        outputs=[status_box, flight_radio, flights_state, meta_state,
                 selection_group, confirm_group],
    )

    confirm_btn.click(
        do_confirm,
        inputs=[flight_radio, flights_state, meta_state],
        outputs=[confirm_output, confirm_group],
    )

if __name__ == "__main__":
    demo.launch()