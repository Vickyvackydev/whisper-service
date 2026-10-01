from typing import List, Dict, Any, Optional
import re

PUNCT_CHARS = '.,!?;:"\'()[]{}'

def apply_proper_case(token: str, proper_word: str) -> str:
    if not token:
        return proper_word
    trimmed = token.strip(PUNCT_CHARS)
    if not trimmed:
        return token
    start_pos = token.find(trimmed)
    if start_pos == -1:
        return proper_word
    prefix = token[:start_pos]
    suffix = token[start_pos + len(trimmed):]
    if proper_word.endswith(".") and suffix.startswith("."):
        suffix = suffix[1:]
    return f"{prefix}{proper_word}{suffix}"

def clean_token(s: str) -> str:
    return str(s).strip(PUNCT_CHARS).lower()

COURT_REPLACEMENTS = [
    # Honorifics & Court Address
    (r'(?i)\bmy\s+lord\b', 'My Lord'),
    (r'(?i)\bmilord\b', 'My Lord'),
    (r'(?i)\bme\s+lord\b', 'My Lord'),
    (r'(?i)\bmy\s+noble\s+lord\b', 'My Noble Lord'),
    (r'(?i)\byour\s+h(?:onou?r)\b', 'Your Honour'),
    (r'(?i)\byour\s+lordship\b', 'Your Lordship'),
    (r'(?i)\byour\s+lordships\b', 'Your Lordships'),
    (r'(?i)\byour\s+ladyship\b', 'Your Ladyship'),
    (r'(?i)\byour\s+ladyships\b', 'Your Ladyships'),
    (r'(?i)\byour\s+worship\b', 'Your Worship'),
    (r'(?i)\byour\s+highness\b', 'Your Highness'),
    (r'(?i)\blearned\s+silk\b', 'Learned Silk'),
    (r'(?i)\blearned\s+friend\b', 'Learned Friend'),
    (r'(?i)\blearned\s+counsel\b', 'Learned Counsel'),
    (r'(?i)\blearned\s+colleague\b', 'Learned Colleague'),
    (r'(?i)\bas\s+the\s+court\s+pleases\b', 'As the Court pleases'),
    (r'(?i)\bas\s+you\s+call\s+places?\b', 'As the Court pleases'),
    (r'(?i)\bcall\s+places?\b', 'Court pleases'),
    (r'(?i)\bmay\s+it\s+please\s+the\s+court\b', 'May it please the Court'),
    (r'(?i)\bsenior\s+advocate\s+of\s+nigeria\b', 'Senior Advocate of Nigeria'),
    (r'(?i)\bmotion\s+on\s+notice\b', 'Motion on Notice'),
    (r'(?i)\bmotion\s+and\s+notice\b', 'Motion on Notice'),
    (r'(?i)\bmotion\s+ex\s+parte\b', 'Motion Ex Parte'),
    (r'(?i)\bright\s+of\s+summons\b', 'writ of summons'),
    (r'(?i)\brate\s+of\s+summons\b', 'writ of summons'),
    (r'(?i)\bfront(?:al)?[\s-]+(?:ended|layer|dead)\s+processes\b', 'front-loaded processes'),
    (r'(?i)\bfront[\s-]+(?:ended|layer|dead)\b', 'front-loaded'),
    (r'(?i)\bsubsets?\s+service\b', 'substituted service'),
    (r'(?i)\binterpleader\s+summon\b', 'interpleader summons'),
    (r'(?i)\b(?:uncom|outcome)\s+proceedings\b', 'ongoing proceedings'),
    (r'(?i)\b(?:Lord\s+)?Justice\s+(?:Seydoux|Seyidu|Seydu)\b', 'Justice Saidu'),
    (r'(?i)\b(?:Seydoux|Seyidu|Seydu)\b', 'Saidu'),
    (r'(?i)\bMr\.?\s+Komolafe\b', 'Mr. Komolafe'),
    (r'(?i)\bKamolafe\b', 'Komolafe'),
    (r'(?i)\badjourned\s+for\s+mentioned\b', 'adjourned for mention'),
    (r'(?i)\bEU\s+health\b', 'ill-health'),
    (r'(?i)\bEU\s+Health\b', 'Ill-health'),
]

PRONOUN_I_REPLACEMENTS = [
    (r'\bi\b', 'I'),
    (r"\bi'm\b", "I'm"),
    (r"\bi've\b", "I've"),
    (r"\bi'll\b", "I'll"),
    (r"\bi'd\b", "I'd"),
]

def normalize_case_numbers_and_slashes(text: str) -> str:
    if not text:
        return text
    
    # 1. Repeatedly resolve slashes between alphanumeric terms (e.g. FHC / L / CS / 485 / 2026 -> FHC/L/CS/485/2026)
    prev = None
    curr = text
    while prev != curr:
        prev = curr
        curr = re.sub(
            r'([A-Za-z0-9\.]+)[\s-]*(?:slash|Slash|\/|\\)[\s-]*([A-Za-z0-9\.]+)',
            r'\1/\2',
            curr
        )
    
    # 2. Handle standalone "-slash-" or "-slash " or " slash-"
    curr = re.sub(r'[\s-]*(?:slash|Slash)[\s-]+', '/', curr)

    # 3. Collapse multiple spaces around remaining slashes if any
    curr = re.sub(r'\s*/\s*', '/', curr)

    # 4. Uppercase slashed suit numbers (e.g. fhc/l/cs/485/2026 -> FHC/L/CS/485/2026)
    curr = re.sub(
        r'\b([A-Za-z0-9\.]+(?:/[A-Za-z0-9\.]+)+)\b',
        lambda m: m.group(1).upper(),
        curr
    )
    
    # 5. Clean up legal misrecognitions & court honorifics
    for pattern, replacement in COURT_REPLACEMENTS:
        curr = re.sub(pattern, replacement, curr)

    # 6. Capitalize standalone pronoun "I" and its common contractions
    for pattern, replacement in PRONOUN_I_REPLACEMENTS:
        curr = re.sub(pattern, replacement, curr)
        
    return curr

def clean_word_token(word: str) -> str:
    if not word:
        return word
    clean = word.strip()
    lower = clean.lower()

    # Normalize slash tokens
    if lower in ("-slash", "slash-", "-slash-", "slash", "\\"):
        return "/"

    # Clean leading/trailing hyphen if attached to number in case number like -21 or -2025
    if re.match(r'^-\d+', clean):
        clean = clean.lstrip('-')

    # Pronoun I normalization
    if lower == "i":
        return "I"
    if lower == "i'm":
        return "I'm"
    if lower == "i've":
        return "I've"
    if lower == "i'll":
        return "I'll"
    if lower == "i'd":
        return "I'd"

    # Known court keywords title-casing when appearing as standalone word tokens
    court_title_map = {
        "milord": "My Lord",
        "lordship": "Lordship",
        "lordships": "Lordships",
        "ladyship": "Ladyship",
        "ladyships": "Ladyships",
        "worship": "Worship",
        "highness": "Highness",
        "honour": "Honour",
        "honor": "Honour",
    }
    if lower in court_title_map:
        return court_title_map[lower]

    # Specific court acronyms
    court_acronyms = {"fhc", "nicn", "san"}
    if lower in court_acronyms:
        return clean.upper()

    return clean

def merge_slashed_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges sequences of word tokens that form slashed suit/case numbers, e.g.:
    ['FHC', '/', 'L', '/', 'CS', '/', '485', '/', '2026'] -> ['FHC/L/CS/485/2026']
    Preserves exact timing from the first token's start to the last token's end.
    """
    if not words:
        return words

    merged: List[Dict[str, Any]] = []
    i = 0
    n = len(words)

    while i < n:
        curr = words[i]
        curr_word = str(curr.get("word", "")).strip()

        # Check if this token or the next token initiates a slash sequence
        is_slash_start = False
        if i + 2 < n and str(words[i+1].get("word", "")).strip() == "/":
            is_slash_start = True
        elif curr_word == "/" and i + 1 < n:
            is_slash_start = True
        elif curr_word.endswith("/") and i + 1 < n:
            is_slash_start = True

        if is_slash_start:
            combined_text = curr_word
            start_time = curr.get("start", 0.0)
            end_time = curr.get("end", 0.0)
            speaker = curr.get("speaker", "SPEAKER_00")
            score = curr.get("score")

            j = i + 1
            while j < n:
                next_item = words[j]
                next_word = str(next_item.get("word", "")).strip()

                if combined_text.endswith("/") or next_word == "/" or next_word.startswith("/"):
                    combined_text = f"{combined_text}{next_word}".replace("//", "/")
                    end_time = next_item.get("end", end_time)
                    j += 1
                elif j + 1 < n and str(words[j+1].get("word", "")).strip() == "/":
                    combined_text = f"{combined_text}/{next_word}".replace("//", "/")
                    end_time = next_item.get("end", end_time)
                    j += 1
                else:
                    break

            # Clean double slashes & uppercase suit numbers
            combined_text = re.sub(r'/+', '/', combined_text).upper()
            merged.append({
                "word": combined_text,
                "start": start_time,
                "end": end_time,
                "score": score,
                "speaker": speaker,
                "source_word": curr.get("source_word"),
                "mapping_type": curr.get("mapping_type")
            })
            i = j
        else:
            merged.append(curr)
            i += 1

    return merged

def normalize_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    if "text" in segment and segment["text"]:
        segment["text"] = normalize_case_numbers_and_slashes(segment["text"])
        
    if "words" in segment and segment["words"]:
        # 1. Clean individual word tokens
        for w in segment["words"]:
            if "word" in w and w["word"]:
                w["word"] = clean_word_token(w["word"])
        
        # 2. Contextual honorific casing for two-word sequences (e.g. "my" + "lord" -> "My" + "Lord")
        words = segment["words"]
        for idx in range(len(words) - 1):
            w1_clean = clean_token(words[idx].get("word", ""))
            w2_clean = clean_token(words[idx+1].get("word", ""))

            if (w1_clean in ("my", "me")) and (w2_clean in ("lord", "noble")):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "My")
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "Lord" if w2_clean == "lord" else "Noble")
            elif w1_clean == "your":
                if w2_clean in ("honour", "honor"):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "Your")
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "Honour")
                elif w2_clean in ("lordship", "lordships", "ladyship", "ladyships", "worship", "highness"):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "Your")
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], w2_clean.capitalize())
            elif w1_clean == "learned" and w2_clean in ("silk", "friend", "counsel", "colleague"):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "Learned")
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], w2_clean.capitalize())
            elif w1_clean in ("mr", "mrs", "justice", "barrister") and len(w2_clean) > 1:
                title = "Mr." if w1_clean == "mr" else ("Mrs." if w1_clean == "mrs" else w1_clean.capitalize())
                words[idx]["word"] = apply_proper_case(words[idx]["word"], title)
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], w2_clean.capitalize())

        # 2b. Legal terminology corrections across words
        for idx in range(len(words)):
            w_clean = clean_token(words[idx].get("word", ""))
            if w_clean == "kamolafe":
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "Komolafe")
            elif w_clean in ("seydoux", "seyidu", "seydu"):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "Saidu")

            if idx + 1 < len(words):
                w1_clean = w_clean
                w2_clean = clean_token(words[idx+1].get("word", ""))
                if w1_clean in ("right", "rate") and w2_clean == "of" and idx + 2 < len(words) and clean_token(words[idx+2].get("word", "")) == "summons":
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "writ")
                elif w1_clean == "front" and w2_clean in ("ended", "-ended", "layer", "-layer", "dead", "-dead"):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "front-loaded")
                    words[idx+1]["word"] = ""
                elif w1_clean in ("front-ended", "front-layer", "front-dead"):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "front-loaded")
                elif w1_clean in ("subsets", "subset") and w2_clean == "service":
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "substituted")
                elif w1_clean == "interpleader" and w2_clean == "summon":
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "summons")
                elif w1_clean in ("uncom", "outcome") and w2_clean == "proceedings":
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "ongoing")
                elif w1_clean == "for" and w2_clean == "mentioned":
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "mention")

        words = [w for w in words if w.get("word")]

        # 3. Merge slash tokens for suit numbers (e.g. FHC / L / CS / 485 / 2026 -> FHC/L/CS/485/2026)
        segment["words"] = merge_slashed_words(words)
                
    return segment
