# app/api/transcript_routes.py
from fastapi import APIRouter, HTTPException
from typing import Optional
from src.services.transcript_storage import transcript_storage

router = APIRouter(prefix="/api/transcripts", tags=["transcripts"])

@router.get("/correlation/{correlation_token}")
async def get_transcript_by_correlation(correlation_token: str):
    """Retrieve transcript by correlation token"""
    transcript = await transcript_storage.get_transcript_by_correlation(correlation_token)
    
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    return transcript.to_dict()

@router.get("/session/{session_id}")
async def get_transcript_by_session(session_id: str):
    """Retrieve transcript by session ID"""
    transcript = await transcript_storage.get_transcript_by_session(session_id)
    
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    return transcript.to_dict()

@router.post("/cleanup")
async def cleanup_old_transcripts(hours: int = 24):
    """Clean up transcripts older than specified hours"""
    await transcript_storage.cleanup_old_transcripts(hours)
    return {"message": f"Cleaned up transcripts older than {hours} hours"}