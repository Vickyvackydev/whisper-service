import re
from typing import List, Dict, Any

def normalize_case_numbers_and_slashes(text: str) -> str:
    if not text:
        return text
    
    # 1. Repeatedly resolve slashes between alphanumeric terms (e.g. E-slash-21-slash-2025 -> E/21/2025)
    prev = None
    curr = text
    while prev != curr:
        prev = curr
        curr = re.sub(
            r'([A-Za-z0-9]+)[\s-]*(?:slash|Slash|\/)[\s-]*([A-Za-z0-9]+)',
            r'\1/\2',
            curr
        )
    
    # 2. Handle standalone "-slash-" or "-slash " or " slash-"
    curr = re.sub(r'[\s-]*(?:slash|Slash)[\s-]+', '/', curr)
    
    # 3. Clean up common legal misrecognitions & court terminology
    legal_replacements = [
        (r'\bEU\s+health\b', 'ill-health'),
        (r'\bEU\s+Health\b', 'Ill-health'),
        (r'\bmilord\b', 'My Lord'),
        (r'\bMilord\b', 'My Lord'),
        (r'\bmy noble lord\b', 'My Noble Lord'),
        (r'\byour lordship\b', 'Your Lordship'),
        (r'\blearned friend\b', 'learned friend'),
        (r'\bmotion and notice\b', 'motion on notice'),
        (r'\bMotion and Notice\b', 'Motion on Notice'),
    ]
    
    for pattern, replacement in legal_replacements:
        curr = re.sub(pattern, replacement, curr)
        
    return curr

def clean_word_token(word: str) -> str:
    if not word:
        return word
    clean = word.strip()
    # Normalize slash tokens
    if clean.lower() in ("-slash", "slash-", "-slash-", "slash"):
        return "/"
    # Clean leading/trailing hyphen if attached to number in case number like -21 or -2025
    if re.match(r'^-\d+', clean):
        return clean.lstrip('-')
    return clean

def normalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    if "text" in segment and segment["text"]:
        segment["text"] = normalize_case_numbers_and_slashes(segment["text"])
        
    if "words" in segment and segment["words"]:
        for w in segment["words"]:
            if "word" in w and w["word"]:
                w["word"] = clean_word_token(w["word"])
                
    return segment
