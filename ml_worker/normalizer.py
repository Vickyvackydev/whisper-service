from typing import List, Dict, Any, Optional
import re

PUNCT_CHARS = '.,!?;:"\'()[]{}'

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19
}

NUM_PATTERN_STR = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?|\d+)'

RATE_OR_TIME_NOUNS = {
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "minute", "minutes", "second", "seconds",
    "dollar", "dollars", "pound", "pounds", "penny", "pennies", "cent", "cents",
    "naira", "share", "shares", "head", "heads", "piece", "pieces"
}

def parse_spoken_number(s: str) -> str:
    s = s.strip().lower()
    if s.isdigit():
        return s
    if s in NUMBER_WORDS:
        return str(NUMBER_WORDS[s])
    parts = re.split(r'[\s-]+', s)
    if len(parts) == 2 and parts[0] in TENS and parts[1] in UNITS:
        return str(TENS[parts[0]] + UNITS[parts[1]])
    return s

def is_number_or_word(s: str) -> bool:
    clean = clean_token(s)
    if clean.isdigit():
        return True
    if clean in NUMBER_WORDS:
        return True
    parts = clean.split('-')
    if len(parts) == 2 and parts[0] in TENS and parts[1] in UNITS:
        return True
    return False

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
    (r'(?i)\bas\s+(?:you\s+call|equal)\s+places?\b', 'As the Court pleases'),
    (r'(?i)\bcall\s+places?\b', 'Court pleases'),
    (r'(?i)\bmay\s+it\s+please\s+the\s+court\b', 'May it please the Court'),
    (r'(?i)\bsenior\s+advocate\s+of\s+nigeria\b', 'Senior Advocate of Nigeria'),
    (r'(?i)\bmotion\s+on\s+notice\b', 'Motion on Notice'),
    (r'(?i)\bmotion\s+and\s+notice\b', 'Motion on Notice'),
    (r'(?i)\bmotion\s+ex\s+parte\b', 'Motion Ex Parte'),
    (r'(?i)\b(?:right|rate)\s+of\s+(?:summons|someone\'?s)\b', 'writ of summons'),
    (r'(?i)\bfrontogect\b', 'front-loaded'),
    (r'(?i)\bfront(?:al)?[\s-]+(?:ended|layer|dead|loaded)\s+processes\b', 'front-loaded processes'),
    (r'(?i)\bfront[\s-]+(?:ended|layer|dead|loaded)\b', 'front-loaded'),
    (r'(?i)\bsubsets?\s+service\b', 'substituted service'),
    (r'(?i)\binterpleader\s+summon\b', 'interpleader summons'),
    (r'(?i)\b(?:uncom|outcome|ancor)\s+proceedings\b', 'ongoing proceedings'),
    (r'(?i)\b(?:Lord\s+)?Justice\s+Said(?:\s+you)?\b', 'Justice Saidu'),
    (r'(?i)\b(?:Lord\s+)?Justice\s+(?:Seydoux|Seyidu|Seydu|Zeydo)\b', 'Justice Saidu'),
    (r'(?i)\b(?:Seydoux|Seyidu|Seydu|Zeydo)\b', 'Saidu'),
    (r'(?i)\bMr\.?\s+Komolafe\b', 'Mr. Komolafe'),
    (r'(?i)\bKamolafe\b', 'Komolafe'),
    (r'(?i)\badjourned\s+for\s+mentioned\b', 'adjourned for mention'),
    (r'(?i)\bEU\s+health\b', 'ill-health'),
    (r'(?i)\bEU\s+Health\b', 'Ill-health'),
    (r'(?i)\bOrder\s+(?:8|eight)\s+rule\s+(?:6|six)\s*d\b', 'Order 8 Rule 6(d)'),
    (r'(?i)\brule\s+(?:6|six)\s*d\b', 'Rule 6(d)'),
    (r'(?i)\b(?:6|six)\s*d\b', '6(d)'),
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
    
    # 0. Normalize suit numbers and court identifiers
    # e.g. "suit number", "suits number", "suit no", "suits no" -> "Suit No."
    curr = re.sub(r'(?i)\b(?:suits?)\s+(?:numbers?|no\.?)\b', 'Suit No.', text)
    curr = re.sub(r'(?i)\bcase\s+(?:numbers?|no\.?)\b', 'Case No.', curr)
    curr = re.sub(r'(?i)\bcharge\s+(?:numbers?|no\.?)\b', 'Charge No.', curr)
    curr = re.sub(r'(?i)\bappeal\s+(?:numbers?|no\.?)\b', 'Appeal No.', curr)
    curr = re.sub(r'(?i)\bmatter\s+(?:numbers?|no\.?)\b', 'Matter No.', curr)

    # 1. Year pronunciations: twenty nineteen -> 2019, twenty twenty-four -> 2024, etc.
    year_map = [
        (r'(?i)\btwenty\s+(?:nineteen|19)\b', '2019'),
        (r'(?i)\btwenty\s+(?:twenty-one|twenty\s+one|21)\b', '2021'),
        (r'(?i)\btwenty\s+(?:twenty-two|twenty\s+two|22)\b', '2022'),
        (r'(?i)\btwenty\s+(?:twenty-three|twenty\s+three|23)\b', '2023'),
        (r'(?i)\btwenty\s+(?:twenty-four|twenty\s+four|24)\b', '2024'),
        (r'(?i)\btwenty\s+(?:twenty-five|twenty\s+five|25)\b', '2025'),
        (r'(?i)\btwenty\s+(?:twenty-six|twenty\s+six|26)\b', '2026'),
        (r'(?i)\btwenty\s+(?:twenty-seven|twenty\s+seven|27)\b', '2027'),
        (r'(?i)\btwenty\s+(?:twenty-eight|twenty\s+eight|28)\b', '2028'),
        (r'(?i)\btwenty\s+(?:twenty-nine|twenty\s+nine|29)\b', '2029'),
        (r'(?i)\btwenty\s+(?:twenty|20)\b', '2020'),
        (r'(?i)\btwenty\s+(?:thirty|30)\b', '2030'),
    ]
    for ym, yr in year_map:
        curr = re.sub(ym, yr, curr)

    # Series of 4 spoken single digits (e.g. "one nine zero four" -> 1904)
    def repl_4digits(m):
        d1 = str(NUMBER_WORDS.get(m.group(1).lower(), m.group(1)))
        d2 = str(NUMBER_WORDS.get(m.group(2).lower(), m.group(2)))
        d3 = str(NUMBER_WORDS.get(m.group(3).lower(), m.group(3)))
        d4 = str(NUMBER_WORDS.get(m.group(4).lower(), m.group(4)))
        return f"{d1}{d2}{d3}{d4}"

    curr = re.sub(
        r'(?i)\b(zero|one|two|three|four|five|six|seven|eight|nine|\d)\s+'
        r'(zero|one|two|three|four|five|six|seven|eight|nine|\d)\s+'
        r'(zero|one|two|three|four|five|six|seven|eight|nine|\d)\s+'
        r'(zero|one|two|three|four|five|six|seven|eight|nine|\d)\b',
        repl_4digits,
        curr
    )

    # 2. Repeatedly resolve slashes between alphanumeric terms (e.g. FHC / L / CS / 485 / 2026 -> FHC/L/CS/485/2026)
    prev = None
    while prev != curr:
        prev = curr
        curr = re.sub(
            r'([A-Za-z0-9\.]+)[\s-]*(?:slash|Slash|\/|\\)[\s-]*([A-Za-z0-9\.]+)',
            r'\1/\2',
            curr
        )
    
    # 3. Handle standalone "-slash-" or "-slash " or " slash-"
    curr = re.sub(r'[\s-]*(?:slash|Slash)[\s-]+', '/', curr)

    # 4. Collapse multiple spaces around remaining slashes if any
    curr = re.sub(r'\s*/\s*', '/', curr)

    # 5. Clean up legal misrecognitions & court honorifics
    for pattern, replacement in COURT_REPLACEMENTS:
        curr = re.sub(pattern, replacement, curr)

    # 6. Number + letter combinations: e.g. "five a" / "5 a" / "five A" / "5 A" / "five-a" -> "5A"
    def repl_num_letter(m):
        num_str = m.group(1)
        letter = m.group(2).upper()
        parsed_num = parse_spoken_number(num_str)
        return f"{parsed_num}{letter}"

    # Letter B-Z: any number followed by optional hyphen/space and letter
    curr = re.sub(
        rf'\b({NUM_PATTERN_STR})[\s-]*([b-zB-Z])\b',
        repl_num_letter,
        curr
    )

    # Letter A: ensure not followed by words like day, week, month, year, time, etc.
    curr = re.sub(
        rf'\b({NUM_PATTERN_STR})[\s-]*([aA])\b(?!\s+(?:day|days|week|weeks|month|months|year|years|time|times|minute|minutes|second|seconds|dollar|dollars|pound|pounds|penny|pennies|cent|cents|naira|share|shares|head|heads|piece|pieces))',
        repl_num_letter,
        curr
    )

    # 7. Spoken numbers after court / legal keywords
    # "Suit No. five" -> "Suit No. 5"
    # "Court five" -> "Court 5"
    # "Order five" -> "Order 5"
    # "Rule six" -> "Rule 6"
    # "Exhibit five" -> "Exhibit 5"
    # "number five" -> "No. 5"
    # "No. five" -> "No. 5"
    def repl_prefix_num(m):
        prefix = m.group(1)
        num_str = m.group(2)
        parsed = parse_spoken_number(num_str)
        if prefix.lower() in ("number", "no"):
            prefix = "No."
        elif prefix.lower() == "court":
            prefix = "Court"
        elif prefix.lower() == "order":
            prefix = "Order"
        elif prefix.lower() == "rule":
            prefix = "Rule"
        elif prefix.lower() == "exhibit":
            prefix = "Exhibit"
        elif prefix.lower() == "room":
            prefix = "Room"
        elif prefix.lower() == "suit":
            prefix = "Suit"
        return f"{prefix} {parsed}"

    curr = re.sub(
        rf'\b(Suit\s+No\.?|Case\s+No\.?|Charge\s+No\.?|Appeal\s+No\.?|Matter\s+No\.?|Court|Room|Order|Rule|Exhibit|No\.?|number|paragraph|section|clause|count|page|item)\s+({NUM_PATTERN_STR})\b',
        repl_prefix_num,
        curr,
        flags=re.IGNORECASE
    )

    # 8. Uppercase slashed suit numbers (e.g. fhc/l/cs/485/2026 -> FHC/L/CS/485/2026)
    # Also resolve any number words inside slashed tokens (e.g. /FIVE/ -> /5/)
    def format_slashed_suit(m):
        raw = m.group(1)
        parts = raw.split('/')
        new_parts = []
        for p in parts:
            p_clean = p.strip()
            if p_clean.lower() in NUMBER_WORDS:
                new_parts.append(str(NUMBER_WORDS[p_clean.lower()]))
            else:
                new_parts.append(p_clean.upper())
        return "/".join(new_parts)

    curr = re.sub(
        r'\b([A-Za-z0-9\.]+(?:/[A-Za-z0-9\.]+)+)\b',
        format_slashed_suit,
        curr
    )

    # 9. Capitalize standalone pronoun "I" and its common contractions
    for pattern, replacement in PRONOUN_I_REPLACEMENTS:
        curr = re.sub(pattern, replacement, curr)
        
    # 10. Normalize 'My' casing: only capitalize in court honorifics ("My Lord", "My Ladyship", etc.) or at sentence start
    curr = normalize_my_casing(curr)

    return curr

COURT_MY_TRAILS = {'lord', 'lords', 'lordship', 'lordships', 'ladyship', 'ladyships', 'noble'}

def normalize_my_casing(text: str) -> str:
    """
    Normalizes 'my' / 'My' casing across text:
    - Court honorifics ('My Lord', 'My Lords', 'My Noble Lord', 'My Ladyship', 'My Lordship') are always capitalized.
    - 'My' at the beginning of a sentence is kept capitalized.
    - Any individual 'My' appearing in the middle of a sentence (e.g. 'this is my client', 'in my view', 'that is my submission')
      is lowercased to 'my'.
    """
    if not text:
        return text

    # 1. First ensure court honorifics with 'my' or 'me' are properly capitalized
    text = re.sub(r'(?i)\b(?:my|me)\s+noble\s+lords?\b', lambda m: 'My Noble Lords' if m.group(0).lower().endswith('s') else 'My Noble Lord', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+lords?\b', lambda m: 'My Lords' if m.group(0).lower().endswith('s') else 'My Lord', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+lordships?\b', lambda m: 'My Lordships' if m.group(0).lower().endswith('s') else 'My Lordship', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+ladyships?\b', lambda m: 'My Ladyships' if m.group(0).lower().endswith('s') else 'My Ladyship', text)

    # 2. Tokenize by words and spaces, keeping punctuation
    tokens = re.split(r'(\s+)', text)
    new_tokens = []
    
    sentence_start = True

    for i, tok in enumerate(tokens):
        if tok.isspace() or not tok:
            new_tokens.append(tok)
            continue

        clean = tok.strip(PUNCT_CHARS)

        if clean == 'My':
            # Check next non-space token
            next_word_clean = ''
            for j in range(i + 1, len(tokens)):
                if not tokens[j].isspace() and tokens[j]:
                    next_word_clean = tokens[j].strip(PUNCT_CHARS).lower()
                    break

            is_court_honorific = next_word_clean in COURT_MY_TRAILS
            if not is_court_honorific and not sentence_start:
                # Replace 'My' with 'my', preserving any punctuation around it
                start_pos = tok.find('My')
                if start_pos != -1:
                    tok = tok[:start_pos] + 'my' + tok[start_pos + 2:]

        new_tokens.append(tok)
        sentence_start = any(tok.rstrip(PUNCT_CHARS).endswith(p) or tok.endswith(p) for p in ('.', '?', '!'))

    return ''.join(new_tokens)

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

    # Clean number+letter combinations like "5a" -> "5A", "5-a" -> "5A"
    clean_no_punct = clean.strip(PUNCT_CHARS)
    m_nl = re.match(r'^(\d+)[-]?([a-zA-Z])$', clean_no_punct)
    if m_nl:
        return apply_proper_case(clean, f"{m_nl.group(1)}{m_nl.group(2).upper()}")
    m_wnl = re.match(r'^([a-zA-Z]+)[-]([a-zA-Z])$', clean_no_punct)
    if m_wnl and is_number_or_word(m_wnl.group(1)):
        return apply_proper_case(clean, f"{parse_spoken_number(m_wnl.group(1))}{m_wnl.group(2).upper()}")

    return clean

def merge_number_letter_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges sequences of word tokens representing numbers + letter suffix:
    ['five', 'a'] -> ['5A']
    ['5', 'a'] -> ['5A']
    ['five', 'b'] -> ['5B']
    ['1', 'a'] -> ['1A']
    Preserves exact timing from the first token's start to the last token's end.
    """
    if not words:
        return words

    merged: List[Dict[str, Any]] = []
    i = 0
    n = len(words)

    while i < n:
        curr = words[i]
        curr_text = str(curr.get("word", "")).strip()
        curr_clean = clean_token(curr_text)

        # Check if curr is a number (e.g. "5", "five") and next is a single letter (e.g. "a", "A", "b")
        if i + 1 < n and is_number_or_word(curr_clean):
            next_item = words[i+1]
            next_text = str(next_item.get("word", "")).strip()
            next_clean = clean_token(next_text)

            is_valid_letter = False
            if len(next_clean) == 1 and next_clean.isalpha():
                if next_clean == "a":
                    # Check next word after 'a'
                    has_rate_word = False
                    if i + 2 < n:
                        after_clean = clean_token(str(words[i+2].get("word", "")))
                        if after_clean in RATE_OR_TIME_NOUNS:
                            has_rate_word = True
                    if not has_rate_word:
                        is_valid_letter = True
                else:
                    is_valid_letter = True

            if is_valid_letter:
                num_digits = parse_spoken_number(curr_clean)
                combined = f"{num_digits}{next_clean.upper()}"
                trail = ""
                for char in reversed(next_text):
                    if char in PUNCT_CHARS:
                        trail = char + trail
                    else:
                        break
                merged.append({
                    "word": f"{combined}{trail}",
                    "start": curr.get("start", 0.0),
                    "end": next_item.get("end", curr.get("end", 0.0)),
                    "speaker": curr.get("speaker", "SPEAKER_00"),
                    "score": curr.get("score"),
                    "source_word": curr.get("source_word"),
                    "mapping_type": curr.get("mapping_type")
                })
                i += 2
                continue

        merged.append(curr)
        i += 1

    return merged

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
        
        # 2. Merge number + letter combinations (e.g. ['five', 'a'] -> ['5A'] or ['5', 'a'] -> ['5A'])
        words = merge_number_letter_words(segment["words"])

        # 3. Contextual honorific casing & court numbers for two-word/three-word sequences
        for idx in range(len(words) - 1):
            w1_clean = clean_token(words[idx].get("word", ""))
            w2_clean = clean_token(words[idx+1].get("word", ""))

            # Judicial honorifics
            if (w1_clean in ("my", "me")) and (w2_clean in ("lord", "lords", "noble", "lordship", "lordships", "ladyship", "ladyships")):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "My")
                if w2_clean == "noble":
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "Noble")
                elif w2_clean in ("lord", "lords"):
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "Lord" if w2_clean == "lord" else "Lords")
                else:
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], w2_clean.capitalize())
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

            # Suit & Court number normalization across word tokens
            if (w1_clean in ("suit", "suits")) and (w2_clean in ("number", "no")):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "Suit")
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "No.")
            elif w1_clean in ("case", "charge", "appeal", "matter") and w2_clean in ("number", "no"):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], w1_clean.capitalize())
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "No.")
            elif w1_clean in ("court", "order", "rule", "exhibit", "room", "paragraph", "section", "clause", "count", "page", "item") and is_number_or_word(w2_clean):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], w1_clean.capitalize())
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], parse_spoken_number(w2_clean))
            elif w1_clean in ("no", "number") and is_number_or_word(w2_clean):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "No.")
                words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], parse_spoken_number(w2_clean))

            # 3-word sequence: "Suit" "No." "five" -> "Suit" "No." "5"
            if idx + 2 < len(words):
                w3_clean = clean_token(words[idx+2].get("word", ""))
                if (w1_clean in ("suit", "suits", "case", "charge", "appeal", "matter")) and (w2_clean in ("no", "number")) and is_number_or_word(w3_clean):
                    words[idx+2]["word"] = apply_proper_case(words[idx+2]["word"], parse_spoken_number(w3_clean))

        # 4. Legal terminology corrections across words
        for idx in range(len(words)):
            w_clean = clean_token(words[idx].get("word", ""))
            if w_clean == "kamolafe":
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "Komolafe")
            elif w_clean in ("seydoux", "seyidu", "seydu", "zeydo"):
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "Saidu")
            elif w_clean == "frontogect":
                words[idx]["word"] = apply_proper_case(words[idx]["word"], "front-loaded")

            if idx + 1 < len(words):
                w1_clean = w_clean
                w2_clean = clean_token(words[idx+1].get("word", ""))
                if w1_clean in ("right", "rate") and w2_clean == "of" and idx + 2 < len(words) and (clean_token(words[idx+2].get("word", "")) == "summons" or clean_token(words[idx+2].get("word", "")).startswith("someone")):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "writ")
                    words[idx+2]["word"] = apply_proper_case(words[idx+2]["word"], "summons")
                elif w1_clean == "front" and (w2_clean in ("ended", "-ended", "layer", "-layer", "dead", "-dead", "loaded", "-loaded")):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "front-loaded")
                    words[idx+1]["word"] = ""
                elif w1_clean in ("front-ended", "front-layer", "front-dead", "front-loaded", "frontloaded"):
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "front-loaded")
                elif w1_clean in ("subsets", "subset") and w2_clean == "service":
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "substituted")
                elif w1_clean == "interpleader" and w2_clean == "summon":
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "summons")
                elif w1_clean in ("uncom", "outcome", "ancor") and w2_clean == "proceedings":
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], "ongoing")
                elif (w1_clean in ("as", "equal") and w2_clean in ("equal", "call") and idx + 2 < len(words) and clean_token(words[idx+2].get("word", "")) in ("places", "place")):
                    words[idx]["word"] = "As"
                    words[idx+1]["word"] = "the"
                    words[idx+2]["word"] = "Court pleases"
                elif w1_clean == "justice" and w2_clean == "said" and idx + 2 < len(words) and clean_token(words[idx+2].get("word", "")) in ("you", "u"):
                    words[idx+1]["word"] = "Saidu"
                    words[idx+2]["word"] = ""
                elif w1_clean == "justice" and w2_clean == "said":
                    words[idx+1]["word"] = "Saidu"
                elif w1_clean == "for" and w2_clean == "mentioned":
                    words[idx+1]["word"] = apply_proper_case(words[idx+1]["word"], "mention")

        words = [w for w in words if w.get("word")]

        # 5. Lowercase individual 'My' that is NOT at start of sentence and NOT part of an honorific
        sentence_start = True
        for idx in range(len(words)):
            raw_w = words[idx].get("word", "")
            clean_w = raw_w.strip(PUNCT_CHARS)
            if clean_w == "My":
                is_honorific = False
                if idx + 1 < len(words):
                    next_clean = clean_token(words[idx+1].get("word", ""))
                    if next_clean in COURT_MY_TRAILS:
                        is_honorific = True
                if not is_honorific and not sentence_start:
                    words[idx]["word"] = apply_proper_case(raw_w, "my")

            sentence_start = any(raw_w.rstrip(PUNCT_CHARS).endswith(p) or raw_w.endswith(p) for p in ('.', '?', '!'))

        # 6. Merge slash tokens for suit numbers (e.g. FHC / L / CS / 485 / 2026 -> FHC/L/CS/485/2026)
        segment["words"] = merge_slashed_words(words)
                
    return segment

