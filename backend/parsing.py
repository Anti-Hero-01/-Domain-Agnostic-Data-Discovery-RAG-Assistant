# backend/parsing.py
from pathlib import Path
import pdfplumber
import pandas as pd

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

def parse_pdf(path):
    text = []
    meta = {}
    with pdfplumber.open(path) as pdf:
        meta['pages'] = len(pdf.pages)
        for p in pdf.pages:
            page_text = p.extract_text() or ''
            text.append(page_text)
    full_text = '\n'.join(text)
    return meta, full_text

def parse_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        t = f.read()
    return {}, t

def parse_excel(path):
    df = pd.read_excel(path, engine='openpyxl')
    # simple conversion: each row -> one line
    text = df.to_csv(index=False)
    return {'rows': len(df)}, text

def parse_file(path):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.pdf':
        meta, text = parse_pdf(path)
    elif suffix in ['.txt']:
        meta, text = parse_txt(path)
    elif suffix in ['.xls', '.xlsx', '.csv']:
        meta, text = parse_excel(path)
    else:
        meta, text = {}, ''
    chunks = chunk_text(text)
    doc_meta = {
        'file_name': p.name,
        'file_path': str(p),
        **meta
    }
    # each chunk can carry metadata later (page, offset)
    return doc_meta, chunks
