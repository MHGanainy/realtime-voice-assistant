"""
Text processing utilities for voice assistant
"""
import re
from typing import List, Tuple


def split_into_speakable_chunks(text: str, 
                               max_chunk_length: int = 200,
                               prefer_punctuation: bool = True) -> List[str]:
    """
    Split text into chunks suitable for TTS processing.
    
    Args:
        text: Input text to split
        max_chunk_length: Maximum length of each chunk
        prefer_punctuation: Try to split at punctuation marks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_length:
        return [text]
    
    chunks = []
    
    if prefer_punctuation:
        # Try to split at sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
    else:
        # Simple word-based splitting
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= max_chunk_length:
                current_chunk = (current_chunk + " " + word).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
    
    return chunks


def estimate_speech_duration(text: str, wpm: int = 150) -> float:
    """
    Estimate how long it will take to speak the given text.
    
    Args:
        text: Text to estimate
        wpm: Words per minute speaking rate
        
    Returns:
        Estimated duration in seconds
    """
    words = len(text.split())
    return (words / wpm) * 60