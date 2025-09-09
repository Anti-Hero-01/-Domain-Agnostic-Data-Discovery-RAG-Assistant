from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.document_processor import DocumentProcessor
from services.google_drive import GoogleDriveService

router = APIRouter()
doc_processor = DocumentProcessor()
drive_service = GoogleDriveService()

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    results = []
    for file in files:
        try:
            # Process the file based on its type
            result = await doc_processor.process_file(file)
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Files processed successfully", "results": results}

@router.post("/google-drive")
async def process_google_drive(folder_id: str):
    """Process files from Google Drive folder"""
    try:
        files = await drive_service.get_files(folder_id)
        results = []
        for file in files:
            result = await doc_processor.process_drive_file(file)
            results.append(result)
        return {"message": "Drive files processed successfully", "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))