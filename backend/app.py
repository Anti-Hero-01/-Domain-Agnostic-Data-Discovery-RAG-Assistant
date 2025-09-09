# backend/app.py
from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from pathlib import Path
from database import Document, get_db, engine, Base

import shutil

app = FastAPI()
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Upload endpoint ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file locally
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Check if file already exists
    existing = db.query(Document).filter(Document.file_name == file.filename).first()
    if existing:
        return {"message": "File already exists", "id": existing.id}

    # Insert new document
    new_doc = Document(
        file_name=file.filename,
        file_path=str(file_path),
        meta={}
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return {"message": "File uploaded", "id": new_doc.id}

# --- List documents ---
@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [{"id": d.id, "file_name": d.file_name} for d in docs]

# --- Ask question (stub) ---
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    # This is a placeholder; connect your embeddings + LLM here
    answer = f"This is a dummy answer to: '{question}'"
    sources = []
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
