import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"  # FastAPI backend

st.set_page_config(page_title="Mini Doc Q&A", layout="centered")
st.title("📄 Mini Doc Q&A with FastAPI")

# --- Session State ---
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# --- File Upload ---
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file and not st.session_state.uploaded:
    # Get existing docs from backend
    existing_docs = requests.get(f"{API_BASE}/documents").json()

    if any(d["file_name"] == uploaded_file.name for d in existing_docs):
        st.warning(f"⚠️ {uploaded_file.name} is already uploaded.")
        st.session_state.uploaded = True
    else:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        res = requests.post(f"{API_BASE}/upload", files=files)

        if res.status_code == 200:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.session_state.uploaded = True
        else:
            st.error("❌ Upload failed")


# --- Ask a Question ---
st.header("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("⚠️ Please enter a question")
    else:
        data = {"question": question}
        res = requests.post(f"{API_BASE}/ask", json=data)

        if res.status_code == 200:
            st.session_state.last_answer = res.json().get("answer", "No answer found")
        else:
            st.session_state.last_answer = "❌ Error while asking question"

# Display the last answer
if st.session_state.last_answer:
    st.success(f"💡 Answer: {st.session_state.last_answer}")

# --- List Uploaded Docs ---
st.header("Uploaded Documents")
if st.button("Show Documents"):
    res = requests.get(f"{API_BASE}/documents")
    if res.status_code == 200:
        docs = res.json()
        if docs:
            st.subheader("📂 Documents:")
            for d in docs:
                st.write(f"- {d['file_name']} (ID: {d['id']})")
        else:
            st.info("No documents uploaded yet")
    else:
        st.error("❌ Could not fetch documents")
