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

EXHIBIT_ID_PATTERN = rf'(?:{NUM_PATTERN_STR}(?:[\s-]?[A-Za-z])?|[A-Za-z]\b|[A-Za-z]{{1,3}}\d+)'

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

def clean_token(s: str) -> str:
    return str(s).strip(PUNCT_CHARS).lower()

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


# ==============================================================================
# CANONICAL COURTROOM REPLACEMENTS & PHRASES
# ==============================================================================
# Rules structure: (pattern, canonical_replacement, is_adaptive)
# - is_adaptive=True: When at sentence start, capitalizes the first character;
#                     otherwise preserves canonical casing (e.g. 'without prejudice').
# - is_adaptive=False: Always preserves the exact canonical casing (e.g. 'Learned Counsel').

COURT_RULES = [
    # --------------------------------------------------------------------------
    # 1. Judicial Decorum, Honorifics & Court Address
    # --------------------------------------------------------------------------
    (r'(?i)\bas\s+the\s+court\s+pleases\b', 'as the Court pleases', True),
    (r'(?i)\bas\s+(?:you\s+call|equal)\s+places?\b', 'as the Court pleases', True),
    (r'(?i)\bcall\s+places?\b', 'Court pleases', False),
    (r'(?i)\bmay\s+it\s+please\s+the\s+court\b', 'may it please the Court', True),
    (r'(?i)\bmay\s+it\s+please\b(?!\s+the\s+court\b)', 'may it please', True),
    (r'(?i)\bmuch\s+obliged\b', 'much obliged', True),
    (r'(?i)\b(?:my|me)\s+learned\s+colleague\b', 'my learned colleague', True),
    (r'(?i)\b(?:my|me)\s+noble\s+lordship\b', 'My Noble Lordship', False),
    (r'(?i)\b(?:my|me)\s+noble\s+lord\b', 'My Noble Lord', False),
    (r'(?i)\bno\s+objection\b', 'No objection', False),
    (r'(?i)\bhonou?rable\s+court\b', 'Honourable Court', False),
    (r'(?i)\byour\s+honor\b', 'Your Honor', False),
    (r'(?i)\byour\s+honour\b', 'Your Honour', False),
    (r'(?i)\byour\s+ladyships\b', 'Your Ladyships', False),
    (r'(?i)\byour\s+ladyship\b', 'Your Ladyship', False),
    (r'(?i)\byour\s+lordships\b', 'Your Lordships', False),
    (r'(?i)\byour\s+lordship\b', 'Your Lordship', False),
    (r'(?i)\byour\s+worship\b', 'Your Worship', False),
    (r'(?i)\byour\s+highness\b', 'Your Highness', False),
    (r'(?i)\blearned\s+silk\b', 'Learned Silk', False),
    (r'(?i)\blearned\s+friend\b', 'Learned Friend', False),
    (r'(?i)\blearned\s+counsel\b', 'Learned Counsel', False),
    (r'(?i)\bsenior\s+advocate\s+of\s+nigeria\b', 'Senior Advocate of Nigeria', False),
    (r'(?i)\bmilord\b', 'My Lord', False),
    (r'(?i)\bme\s+lord\b', 'My Lord', False),

    # --------------------------------------------------------------------------
    # 2. Judicial Officers & Court Personnel
    # --------------------------------------------------------------------------
    (r'(?i)\bdirector\s+of\s+public\s+prosecutions\b', 'Director of Public Prosecutions', False),
    (r'(?i)\battorney\s+general\b', 'Attorney General', False),
    (r'(?i)\bsolicitor\s+general\b', 'Solicitor General', False),
    (r'(?i)\bchief\s+justice\b', 'Chief Justice', False),
    (r'(?i)\bchief\s+judge\b', 'Chief Judge', False),
    (r'(?i)\bpresiding\s+judge\b', 'Presiding Judge', False),
    (r'(?i)\bsenior\s+judge\b', 'Senior Judge', False),
    (r'(?i)\bchief\s+registrar\b', 'Chief Registrar', False),
    (r'(?i)\bdeputy\s+registrar\b', 'Deputy Registrar', False),
    (r'(?i)\bdeputy\s+sheriff\b', 'Deputy Sheriff', False),
    (r'(?i)\blegal\s+practitioner\b', 'legal practitioner', False),
    (r'(?i)\bsolicitor\b(?!\s+general)', 'solicitor', False),

    # --------------------------------------------------------------------------
    # 3. Parties to a Suit
    # --------------------------------------------------------------------------
    (r'(?i)\bdefence\s+witness\b', 'Defence Witness', False),
    (r'(?i)\bdefense\s+witness\b', 'Defence Witness', False),
    (r'(?i)\bprosecution\s+witness\b', 'Prosecution Witness', False),
    (r'(?i)\bjudgment\s+creditor\b', 'Judgment Creditor', False),
    (r'(?i)\bjudgment\s+debtor\b', 'Judgment Debtor', False),
    (r'(?i)\bcounter[\s-]+claimant\b', 'Counter-Claimant', False),
    (r'(?i)\bco[\s-]+defendant\b', 'Co-Defendant', False),
    (r'(?i)\bco[\s-]+plaintiff\b', 'Co-Plaintiff', False),
    (r'(?i)\bco[\s-]+respondent\b', 'Co-Respondent', False),

    # --------------------------------------------------------------------------
    # 4. Courts & Judicial Bodies
    # --------------------------------------------------------------------------
    (r'(?i)\bnational\s+industrial\s+court\b', 'National Industrial Court', False),
    (r'(?i)\bfederal\s+high\s+court\b', 'Federal High Court', False),
    (r'(?i)\bstate\s+high\s+court\b', 'State High Court', False),
    (r'(?i)\bcourt\s+of\s+appeal\b', 'Court of Appeal', False),
    (r'(?i)\bcourt\s+of\s+arbitration\b', 'Court of Arbitration', False),
    (r'(?i)\bappellate\s+court\b', 'Appellate Court', False),
    (r'(?i)\bcustomary\s+court\b', 'Customary Court', False),
    (r'(?i)\bindustrial\s+court\b', 'Industrial Court', False),
    (r'(?i)\blower\s+court\b', 'Lower Court', False),
    (r'(?i)\bmagistrate(?:\'s)?\s+court\b', 'Magistrate Court', False),
    (r'(?i)\bmatrimonial\s+court\b', 'Matrimonial Court', False),
    (r'(?i)\bsupreme\s+court\b', 'Supreme Court', False),
    (r'(?i)\bvacation\s+court\b', 'Vacation Court', False),
    (r'(?i)\bhigh\s+court\b', 'High Court', False),
    (r'(?i)\bin\s+chambers\b', 'In Chambers', False),
    (r'(?i)\b(?:at|the)\s+bar\b', lambda m: m.group(0).split()[0] + ' Bar', False),
    (r'(?i)\b(?:on|the)\s+bench\b', lambda m: m.group(0).split()[0] + ' Bench', False),

    # --------------------------------------------------------------------------
    # 5. Pleadings, Motions & Documents
    # --------------------------------------------------------------------------
    (r'(?i)\baffidavit\s+of\s+service\b', 'Affidavit of Service', False),
    (r'(?i)\bcounter[\s-]+affidavit\b', 'Counter-Affidavit', False),
    (r'(?i)\bfurther\s+affidavit\b', 'Further Affidavit', False),
    (r'(?i)\bappellant(?:\'s)?\s+brief\b', "Appellant's Brief", False),
    (r'(?i)\brespondent(?:\'s)?\s+brief\b', "Respondent's Brief", False),
    (r'(?i)\bmemorandum\s+of\s+appeal\b', 'Memorandum of Appeal', False),
    (r'(?i)\bnotice\s+of\s+appeal\b', 'Notice of Appeal', False),
    (r'(?i)\boriginating\s+motion\b', 'Originating Motion', False),
    (r'(?i)\boriginating\s+service\b', 'Originating Service', False),
    (r'(?i)\boriginating\s+summons\b', 'Originating Summons', False),
    (r'(?i)\bpower\s+of\s+attorney\b', 'Power of Attorney', False),
    (r'(?i)\breport\s+of\s+service\b', 'Report of Service', False),
    (r'(?i)\breport\s+of\s+settlement\b', 'Report of Settlement', False),
    (r'(?i)\bstatement\s+of\s+claim\b', 'Statement of Claim', False),
    (r'(?i)\bstatement\s+of\s+defen[sc]e\b', 'Statement of Defence', False),
    (r'(?i)\bstatement\s+on\s+oath\b', 'Statement on Oath', False),
    (r'(?i)\bstatutory\s+notice\b', 'Statutory Notice', False),
    (r'(?i)\b(?:writ|right|rate)\s+of\s+(?:summons|someone\'?s)\b', 'Writ of Summons', False),
    (r'(?i)\bwritten\s+address\b', 'Written Address', False),
    (r'(?i)\bcause\s+list\b', 'Cause List', False),
    (r'(?i)\bcounter[\s-]+claim\b', 'Counterclaim', False),
    (r'(?i)\brecord\s+of\s+proceedings\b', 'record of proceedings', False),
    (r'(?i)\bsubpoena\b', 'subpoena', False),
    (r'(?i)\bgazette\b', 'gazette', False),

    # --------------------------------------------------------------------------
    # 6. Orders, Injunctions & Judgments
    # --------------------------------------------------------------------------
    (r'(?i)\binterlocutory\s+injunction\b', 'Interlocutory Injunction', False),
    (r'(?i)\bconsent\s+judgment\b', 'Consent Judgment', False),
    (r'(?i)\bconsent\s+order\b', 'Consent Order', False),
    (r'(?i)\bdeclaratory\s+relief\b', 'Declaratory Relief', False),
    (r'(?i)\bdefault\s+judgment\b', 'Default Judgment', False),
    (r'(?i)\bgarnishee\s+order\b', 'Garnishee Order', False),
    (r'(?i)\bmotion\s+ex\s+parte\b', 'Motion Ex Parte', False),
    (r'(?i)\bmotion\s+expatate\b', 'Motion Ex Parte', False),
    (r'(?i)\bexpatate\b', 'ex parte', False),
    (r'(?i)\bmotion\s+(?:on|and)\s+notice\b', 'Motion on Notice', False),
    (r'(?i)\bcriminal\s+code\s+law\b', 'Criminal Code Law', False),
    (r'(?i)\bcriminal\s+code\b', 'Criminal Code', False),
    (r'(?i)\bpenal\s+code\s+law\b', 'Penal Code Law', False),
    (r'(?i)\bpenal\s+code\b', 'Penal Code', False),
    (r'(?i)\bobjection,?\s+(?:my|me)\s+lord\b', 'objection, my Lord', True),
    (r'(?i)\bakwa\s+ibom\b', 'Akwa Ibom', False),
    (r'(?i)\bcross\s+river\b', 'Cross River', False),
    (r'(?i)\bfederal\s+capital\s+territory\b', 'Federal Capital Territory', False),
    (r'(?i)\border\s+of\s+court\b', 'Order of Court', False),
    (r'(?i)\bperpetual\s+injunction\b', 'Perpetual Injunction', False),
    (r'(?i)\bremand\s+order\b', 'Remand Order', False),
    (r'(?i)\bstay\s+of\s+execution\b', 'Stay of Execution', False),
    (r'(?i)\bstay\s+of\s+proceedings\b', 'Stay of Proceedings', False),
    (r'(?i)\bsummary\s+judgment\b', 'Summary Judgment', False),
    (r'(?i)\binterlocutory\b(?!\s+injunction\b)', 'interlocutory', False),
    (r'(?i)\bordered\s+as\s+prayed\b', 'ordered as prayed', True),

    # --------------------------------------------------------------------------
    # 7. Court Proceedings & Trial Terms
    # --------------------------------------------------------------------------
    (r'(?i)\bcertified\s+true\s+copy\b', 'Certified True Copy', False),
    (r'(?i)\bexamination[\s-]+in[\s-]+chief\b', 'Examination-in-Chief', False),
    (r'(?i)\bcross[\s-]+examination\b', 'Cross-Examination', False),
    (r'(?i)\bcross[\s-]+examine\b', 'cross-examine', False),
    (r'(?i)\bre[\s-]+examination\b', 'Re-examination', False),
    (r'(?i)\bpre[\s-]+trial\b', 'Pre-trial', False),
    (r'(?i)\bpart[\s-]+heard\b', 'part-heard', False),
    (r'(?i)\bfrontogect\b', 'front-loaded', False),
    (r'(?i)\bfront(?:al)?[\s-]+(?:ended|layer|dead|loaded)\s+processes\b', 'front-loaded processes', False),
    (r'(?i)\bfront[\s-]+(?:ended|layer|dead|loaded)\b', 'front-loaded', False),
    (r'(?i)\bfront[\s-]+load\b', 'front-load', False),
    (r'(?i)\b(?:substituted|subsets?)\s+service\b', 'substituted service', False),
    (r'(?i)\badjourned\s+for\s+mentioned\b', 'adjourned for mention', False),
    (r'(?i)\badjourned\s+date\b', 'adjourned date', False),
    (r'(?i)\badjournment\b', 'adjournment', False),
    (r'(?i)\baffixture\b', 'affixture', False),
    (r'(?i)\bappearances\b', 'appearances', False),
    (r'(?i)\bappearance\b', 'appearance', False),
    (r'(?i)\badmissibility\b', 'admissibility', False),
    (r'(?i)\bonus\b', 'onus', False),
    (r'(?i)\boverruled\b', 'overruled', False),
    (r'(?i)\bruling\b', 'Ruling', False),
    (r'(?i)\binterpleader\s+summon\b', 'interpleader summons', False),
    (r'(?i)\b(?:uncom|outcome|ancor)\s+proceedings\b', 'ongoing proceedings', False),

    # --------------------------------------------------------------------------
    # 8. Latin Maxims & Legal Doctrines
    # --------------------------------------------------------------------------
    (r'(?i)\bex[\s-]+parte\b', 'Ex Parte', False),
    (r'(?i)\bin[\s-]+limine\b', 'In Limine', False),
    (r'(?i)\bper\s+se\b', 'Per Se', False),
    (r'(?i)\ballocutus\b', 'allocutus', False),
    (r'(?i)\bestoppel\b', 'estoppel', False),
    (r'(?i)\bstatute[\s-]+barred\b', 'statute-barred', False),
    (r'(?i)\binter[\s-]+alia\b', 'inter alia', False),
    (r'(?i)\blocus\s+in\s+quo\b', 'locus in quo', False),
    (r'(?i)\blocus\s+standi\b', 'locus standi', False),
    (r'(?i)\bmutatis\s+mutandis\b', 'mutatis mutandis', False),
    (r'(?i)\bobiter\s+dictum\b', 'obiter dictum', False),
    (r'(?i)\bprima\s+facie\b', 'prima facie', False),
    (r'(?i)\bratio\s+decidendi\b', 'ratio decidendi', False),
    (r'(?i)\bres\s+judicata\b', 'res judicata', False),
    (r'(?i)\bsub[\s-]?judice\b', 'subjudice', False),
    (r'(?i)\bsuo\s+mot(?:ou|u)\b', 'suo motou', False),
    (r'(?i)\bultra\s+vires\b', 'ultra vires', False),

    # --------------------------------------------------------------------------
    # 9. Formal Connectives
    # --------------------------------------------------------------------------
    (r'(?i)\baforementioned\b', 'aforementioned', False),
    (r'(?i)\baforesaid\b', 'aforesaid', False),
    (r'(?i)\bwhereof\b', 'Whereof', False),
    (r'(?i)\bwith\s+due\s+respect\b', 'with due respect', True),
    (r'(?i)\bwithout\s+prejudice\b', 'without prejudice', True),

    # --------------------------------------------------------------------------
    # 10. Specific Names & Common Nigerian Court Entities
    # --------------------------------------------------------------------------
    (r'(?i)\b(?:Lord\s+)?Justice\s+Said(?:\s+you)?\b', 'Justice Saidu', False),
    (r'(?i)\b(?:Lord\s+)?Justice\s+(?:Seydoux|Seyidu|Seydu|Zeydo)\b', 'Justice Saidu', False),
    (r'(?i)\b(?:Seydoux|Seyidu|Seydu|Zeydo)\b', 'Saidu', False),
    (r'(?i)\bMr\.?\s+Komolafe\b', 'Mr. Komolafe', False),
    (r'(?i)\bKamolafe\b', 'Komolafe', False),
    (r'(?i)\bEU\s+health\b', 'ill-health', True),
    (r'(?i)\bOrder\s+(?:8|eight)\s+rule\s+(?:6|six)\s*d\b', 'Order 8 Rule 6(d)', False),
    (r'(?i)\brule\s+(?:6|six)\s*d\b', 'Rule 6(d)', False),
    (r'(?i)\b(?:6|six)\s*d\b', '6(d)', False),
]

# Standalone single-word court roles and entities that should be capitalized
COURT_TITLE_WORDS = {
    "accused": "Accused",
    "appellant": "Appellant",
    "appellee": "Appellee",
    "applicant": "Applicant",
    "claimant": "Claimant",
    "defendant": "Defendant",
    "deponent": "Deponent",
    "garnishee": "Garnishee",
    "petitioner": "Petitioner",
    "plaintiff": "Plaintiff",
    "prosecution": "Prosecution",
    "respondent": "Respondent",
    "surety": "Surety",
    "witness": "Witness",
    "bailiff": "Bailiff",
    "barrister": "Barrister",
    "clerk": "Clerk",
    "counsel": "Counsel",
    "judge": "Judge",
    "justice": "Justice",
    "magistrate": "Magistrate",
    "registrar": "Registrar",
    "sheriff": "Sheriff",
    "chamber": "Chamber",
    "courtroom": "Courtroom",
    "crown": "Crown",
    "registry": "Registry",
    "tribunal": "Tribunal",
    "affidavit": "Affidavit",
    "summons": "Summons",
    "writ": "Writ",
    "ruling": "Ruling",
    "whereof": "Whereof",
    "milord": "My Lord",
    "lord": "Lord",
    "lords": "Lords",
    "lordship": "Lordship",
    "lordships": "Lordships",
    "ladyship": "Ladyship",
    "ladyships": "Ladyships",
    "worship": "Worship",
    "highness": "Highness",
    "honour": "Honour",
    "honor": "Honour",
}

PRONOUN_I_REPLACEMENTS = [
    (r'\bi\b', 'I'),
    (r"\bi'm\b", "I'm"),
    (r"\bi've\b", "I've"),
    (r"\bi'll\b", "I'll"),
    (r"\bi'd\b", "I'd"),
]

CALENDAR_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos",
    "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers",
    "Sokoto", "Taraba", "Yobe", "Zamfara", "Federal Capital Territory", "Abuja"
]

CALENDAR_MONTHS_MAP = {m.lower(): m for m in CALENDAR_MONTHS}
NIGERIAN_STATES_MAP = {s.lower(): s for s in NIGERIAN_STATES}

def apply_court_rules(text: str) -> str:
    """
    Applies courtroom phrase replacements with sentence-start awareness for conversational phrases.
    """
    for item in COURT_RULES:
        pattern = item[0]
        replacement = item[1]
        is_adaptive = item[2] if len(item) > 2 else False

        if callable(replacement):
            text = re.sub(pattern, replacement, text)
        elif is_adaptive:
            def repl_adaptive(m, rep=replacement):
                start_idx = m.start()
                prefix = text[:start_idx].rstrip()
                is_start = not prefix or prefix[-1] in ('.', '!', '?', ':', '\n')
                if is_start and rep:
                    return rep[0].upper() + rep[1:]
                return rep
            text = re.sub(pattern, repl_adaptive, text)
        else:
            text = re.sub(pattern, replacement, text)

    return text

def normalize_case_numbers_and_slashes(text: str) -> str:
    if not text:
        return text

    # --------------------------------------------------------------------------
    # 0. Normalize suit numbers and court identifiers
    # --------------------------------------------------------------------------
    curr = re.sub(r'(?i)\b(?:suits?)\s+(?:numbers?|no\.?)(?!\w)', 'Suit No.', text)
    curr = re.sub(r'(?i)\bcase\s+(?:numbers?|no\.?)(?!\w)', 'Case No.', curr)
    curr = re.sub(r'(?i)\bcharge\s+(?:numbers?|no\.?)(?!\w)', 'Charge No.', curr)
    curr = re.sub(r'(?i)\bappeal\s+(?:numbers?|no\.?)(?!\w)', 'Appeal No.', curr)
    curr = re.sub(r'(?i)\bmatter\s+(?:numbers?|no\.?)(?!\w)', 'Matter No.', curr)

    # --------------------------------------------------------------------------
    # 1. Year pronunciations: twenty nineteen -> 2019, twenty twenty-four -> 2024, etc.
    # --------------------------------------------------------------------------
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

    # Convert hyphenated suit numbers to slashes (e.g. E-57D-2023 -> E/57D/2023, E-447M-2023 -> E/447M/2023)
    curr = re.sub(
        r'\b([A-Za-z]{1,6})-([A-Za-z]{1,6})-([A-Za-z]{1,6})-(\d+[A-Za-z]?)-((?:19|20)\d{2})\b',
        lambda m: f"{m.group(1).upper()}/{m.group(2).upper()}/{m.group(3).upper()}/{m.group(4).upper()}/{m.group(5)}",
        curr
    )
    curr = re.sub(
        r'\b([A-Za-z]{1,6})-([A-Za-z]{1,6})-(\d+[A-Za-z]?)-((?:19|20)\d{2})\b',
        lambda m: f"{m.group(1).upper()}/{m.group(2).upper()}/{m.group(3).upper()}/{m.group(4)}",
        curr
    )
    curr = re.sub(
        r'\b([A-Za-z]{1,6})-(\d+[A-Za-z]?)-((?:19|20)\d{2})\b',
        lambda m: f"{m.group(1).upper()}/{m.group(2).upper()}/{m.group(3)}",
        curr
    )
    curr = re.sub(
        r'(?i)\b(Suit\s+No\.?|Case\s+No\.?|Charge\s+No\.?|Appeal\s+No\.?|Matter\s+No\.?|Motion\s+(?:Ex\s+Parte|on\s+Notice))\s+([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+(?:-(?:19|20)\d{2}))\b',
        lambda m: f"{m.group(1)} {m.group(2).replace('-', '/').upper()}",
        curr
    )

    # --------------------------------------------------------------------------
    # 2. Slashes: "The word slash should appear as / (no space before and after)"
    # --------------------------------------------------------------------------
    # Replace the word "slash" (with optional dashes or spaces around it) with '/'
    curr = re.sub(r'[\s-]*(?i:\bslash\b)[\s-]*', '/', curr)

    # Repeatedly resolve slashes between alphanumeric terms (e.g. FHC / L / CS / 485 / 2026 -> FHC/L/CS/485/2026)
    prev = None
    while prev != curr:
        prev = curr
        curr = re.sub(
            r'([A-Za-z0-9\.]+)[\s-]*(?:/|\\)[\s-]*([A-Za-z0-9\.]+)',
            r'\1/\2',
            curr
        )

    # Clean any remaining spaces around slashes: no space before and after
    curr = re.sub(r'\s*/\s*', '/', curr)
    curr = re.sub(r'/+', '/', curr)

    # --------------------------------------------------------------------------
    # 3. Format Slashed Tokens (suit numbers vs regular words)
    # --------------------------------------------------------------------------
    def format_slashed_token(m):
        raw = m.group(1)
        parts = raw.split('/')
        new_parts = []
        has_digits = any(p.strip().isdigit() or p.strip().lower() in NUMBER_WORDS for p in parts)
        has_suit_acronyms = any(p.strip().lower() in ('fhc', 'nicn', 'ca', 'sc', 'cs', 'cr', 'abj', 'l', 'm', 'cv') for p in parts)
        is_suit_format = has_digits or has_suit_acronyms

        for p in parts:
            p_clean = p.strip()
            p_lower = p_clean.lower()
            if p_lower in NUMBER_WORDS:
                new_parts.append(str(NUMBER_WORDS[p_lower]))
            elif p_lower in COURT_TITLE_WORDS:
                new_parts.append(COURT_TITLE_WORDS[p_lower])
            elif is_suit_format and (len(p_clean) <= 4 or p_clean.isupper()):
                new_parts.append(p_clean.upper())
            else:
                new_parts.append(p_clean)
        return "/".join(new_parts)

    curr = re.sub(
        r'\b([A-Za-z0-9\.]+(?:/[A-Za-z0-9\.]+)+)\b',
        format_slashed_token,
        curr
    )

    # --------------------------------------------------------------------------
    # 4. Canonical Courtroom Rules & Terminology
    # --------------------------------------------------------------------------
    curr = apply_court_rules(curr)

    # Calendar Months capitalization
    for mon in CALENDAR_MONTHS:
        if mon != "May":
            curr = re.sub(rf'(?i)\b{mon}\b', mon, curr)

    # Month 'May': only capitalize in date/calendar contexts to avoid collision with modal verb 'may'
    curr = re.sub(
        r'(?i)\b(?:(?:\d{1,2}(?:st|nd|rd|th)?\s+)?of\s+|in\s+|since\s+|by\s+|until\s+|during\s+|on\s+|before\s+|after\s+)may\b',
        lambda m: m.group(0)[:-3] + "May",
        curr
    )
    curr = re.sub(
        r'(?i)\bmay(?=\s+(?:\d{1,2}(?:st|nd|rd|th)?|\d{4})\b)',
        'May',
        curr
    )

    # Single-word Nigerian States capitalization
    for st in NIGERIAN_STATES:
        if " " not in st:
            curr = re.sub(rf'(?i)\b{st}\b', st, curr)

    # --------------------------------------------------------------------------
    # 5. Number + Letter Combinations: e.g. "five a" / "5 a" -> "5A"
    # --------------------------------------------------------------------------
    def repl_num_letter(m):
        num_str = m.group(1)
        letter = m.group(2).upper()
        parsed_num = parse_spoken_number(num_str)
        return f"{parsed_num}{letter}"

    # Letter B-Z
    curr = re.sub(
        rf'\b({NUM_PATTERN_STR})[\s-]*([b-zB-Z])\b',
        repl_num_letter,
        curr
    )

    # Letter A (avoid collision with duration/rate words)
    curr = re.sub(
        rf'\b({NUM_PATTERN_STR})[\s-]*([aA])\b(?!\s+(?:day|days|week|weeks|month|months|year|years|time|times|minute|minutes|second|seconds|dollar|dollars|pound|pounds|penny|pennies|cent|cents|naira|share|shares|head|heads|piece|pieces))',
        repl_num_letter,
        curr
    )

    # --------------------------------------------------------------------------
    # 6. Spoken numbers after court / legal keywords
    # --------------------------------------------------------------------------
    def repl_prefix_num(m):
        prefix = m.group(1)
        num_str = m.group(2)
        parsed = parse_spoken_number(num_str)
        prefix_lower = prefix.lower().strip()
        if prefix_lower in ("number", "no", "no."):
            prefix = "No."
        elif prefix_lower == "court":
            prefix = "Court"
        elif prefix_lower == "order":
            prefix = "Order"
        elif prefix_lower == "rule":
            prefix = "Rule"
        elif prefix_lower in ("exhibit", "exhibits"):
            prefix = "Exhibit" if prefix_lower == "exhibit" else "Exhibits"
        elif prefix_lower == "room":
            prefix = "Room"
        elif prefix_lower in ("suit", "suit no."):
            prefix = "Suit No."
        elif prefix_lower.startswith("suit"):
            prefix = "Suit No."
        elif prefix_lower.startswith("case"):
            prefix = "Case No."
        elif prefix_lower.startswith("charge"):
            prefix = "Charge No."
        elif prefix_lower.startswith("appeal"):
            prefix = "Appeal No."
        elif prefix_lower.startswith("matter"):
            prefix = "Matter No."
        return f"{prefix} {parsed}"

    curr = re.sub(
        rf'\b(Suit\s+No\.?|Case\s+No\.?|Charge\s+No\.?|Appeal\s+No\.?|Matter\s+No\.?|Court|Room|Order|Rule|Exhibit|Exhibits|No\.?|number|paragraph|section|clause|count|page|item)\s+({NUM_PATTERN_STR})\b',
        repl_prefix_num,
        curr,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------------------------
    # 7. Exhibit & Exhibits capitalization rules
    # "Exhibit (when a number follows it, it should be capital E)"
    # "Exhibits (when numbers follow it, it should be capital E)"
    # --------------------------------------------------------------------------
    curr = re.sub(
        rf'(?i)\bexhibits\b(?=\s+{EXHIBIT_ID_PATTERN})',
        'Exhibits',
        curr
    )
    curr = re.sub(
        rf'(?i)\bexhibit\b(?=\s+{EXHIBIT_ID_PATTERN})',
        'Exhibit',
        curr
    )

    # Handle subsequent numbers in Exhibits sequences (e.g. Exhibits 1 and two -> Exhibits 1 and 2)
    curr = re.sub(
        rf'\b(Exhibits\s+(?:\d+|[A-Za-z0-9]+)\s+(?:and|to|through)\s+)({NUM_PATTERN_STR})\b',
        lambda m: f"{m.group(1)}{parse_spoken_number(m.group(2))}",
        curr,
        flags=re.IGNORECASE
    )

    # If exhibit/exhibits is NOT followed by a number or exhibit identifier,
    # lowercase it mid-sentence (capitalize at sentence start)
    def repl_exhibit_no_num(m):
        tok = m.group(1)
        start_idx = m.start()
        prefix = curr[:start_idx].rstrip()
        is_start = not prefix or prefix[-1] in ('.', '!', '?', ':', '\n')
        if is_start:
            return tok[0].upper() + tok[1:].lower()
        return tok.lower()

    curr = re.sub(
        rf'(?i)\b(exhibits?)\b(?!\s+{EXHIBIT_ID_PATTERN})',
        repl_exhibit_no_num,
        curr
    )

    # --------------------------------------------------------------------------
    # 8. Capitalize Standalone Pronoun "I" and contractions
    # --------------------------------------------------------------------------
    for pattern, replacement in PRONOUN_I_REPLACEMENTS:
        curr = re.sub(pattern, replacement, curr)

    # --------------------------------------------------------------------------
    # 9. Normalize 'My' casing: only capitalize in court honorifics or sentence start
    # --------------------------------------------------------------------------
    curr = normalize_my_casing(curr)

    return curr


COURT_MY_TRAILS = {'lord', 'lords', 'lordship', 'lordships', 'ladyship', 'ladyships'}

def normalize_my_casing(text: str) -> str:
    """
    Normalizes 'my' / 'My' casing across text:
    - Court honorifics ('My Lord', 'My Lords', 'My Ladyship', 'My Lordship') are capitalized.
    - 'My Noble Lord', 'My Noble Lordship', 'My learned colleague' are capitalized at sentence start, lowercase 'my' mid-sentence.
    - 'My' at the beginning of a sentence is kept capitalized.
    - Any individual 'My' appearing in the middle of a sentence is lowercased to 'my'.
    """
    if not text:
        return text

    # First ensure court honorifics with 'my' or 'me' are properly formatted
    text = re.sub(r'(?i)\b(?:my|me)\s+noble\s+lordships?\b', lambda m: 'My Noble Lordships' if m.group(0).lower().endswith('s') else 'My Noble Lordship', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+noble\s+lords?\b', lambda m: 'My Noble Lords' if m.group(0).lower().endswith('s') else 'My Noble Lord', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+lords?\b', lambda m: 'My Lords' if m.group(0).lower().endswith('s') else 'My Lord', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+lordships?\b', lambda m: 'My Lordships' if m.group(0).lower().endswith('s') else 'My Lordship', text)
    text = re.sub(r'(?i)\b(?:my|me)\s+ladyships?\b', lambda m: 'My Ladyships' if m.group(0).lower().endswith('s') else 'My Ladyship', text)

    # Tokenize by words and spaces, keeping punctuation
    tokens = re.split(r'(\s+)', text)
    new_tokens = []
    sentence_start = True

    for i, tok in enumerate(tokens):
        if tok.isspace() or not tok:
            new_tokens.append(tok)
            continue

        clean = tok.strip(PUNCT_CHARS)

        next_word_clean = ''
        for j in range(i + 1, len(tokens)):
            if not tokens[j].isspace() and tokens[j]:
                next_word_clean = tokens[j].strip(PUNCT_CHARS).lower()
                break

        prev_word_clean = ''
        for k in range(i - 1, -1, -1):
            if not tokens[k].isspace() and tokens[k]:
                prev_word_clean = tokens[k].strip(PUNCT_CHARS).lower()
                break

        if clean.lower() == 'my':
            if prev_word_clean == 'objection':
                # User rule: objection, my Lord (never objection, My Lord)
                tok = apply_proper_case(tok, 'my')
            elif next_word_clean in COURT_MY_TRAILS or next_word_clean == 'noble':
                tok = apply_proper_case(tok, 'My')
            elif next_word_clean == 'learned':
                casing = 'My' if sentence_start else 'my'
                tok = apply_proper_case(tok, casing)
            elif not sentence_start:
                tok = apply_proper_case(tok, 'my')

        # User rule: mid-sentence 'demand' or 'demands' must be lowercased
        if clean.lower() in ('demand', 'demands') and not sentence_start:
            tok = apply_proper_case(tok, clean.lower())

        # User rule: 'objection' casing based on sentenceStart
        if clean.lower() == 'objection':
            casing = 'Objection' if sentence_start else 'objection'
            tok = apply_proper_case(tok, casing)

        new_tokens.append(tok)
        sentence_start = any(tok.rstrip(PUNCT_CHARS).endswith(p) or tok.endswith(p) for p in ('.', '?', '!'))

    return ''.join(new_tokens)


def clean_word_token(word: str) -> str:
    """
    Cleans and standardizes an individual word token.
    """
    if not word:
        return word
    clean = word.strip()
    lower = clean.lower()

    # Normalize slash tokens to '/'
    if lower in ("-slash", "slash-", "-slash-", "slash", "\\"):
        return "/"

    # Clean leading/trailing hyphen if attached to number in case number like -21 or -2025
    if re.match(r'^-\d+', clean):
        clean = clean.lstrip('-')

    # Hyphenated suit number tokens (e.g. E-57D-2023 -> E/57D/2023, E-447M-2023 -> E/447M/2023)
    m_suit3 = re.match(r'(?i)^([a-zA-Z]{1,6})-(\d+[a-zA-Z]?)-((?:19|20)\d{2})([.,!?;:]*)$', clean)
    if m_suit3:
        return f"{m_suit3.group(1).upper()}/{m_suit3.group(2).upper()}/{m_suit3.group(3)}{m_suit3.group(4)}"
    m_suit4 = re.match(r'(?i)^([a-zA-Z]{1,6})-([a-zA-Z]{1,6})-(\d+[a-zA-Z]?)-((?:19|20)\d{2})([.,!?;:]*)$', clean)
    if m_suit4:
        return f"{m_suit4.group(1).upper()}/{m_suit4.group(2).upper()}/{m_suit4.group(3).upper()}/{m_suit4.group(4)}{m_suit4.group(5)}"
    m_suit5 = re.match(r'(?i)^([a-zA-Z]{1,6})-([a-zA-Z]{1,6})-([a-zA-Z]{1,6})-(\d+[a-zA-Z]?)-((?:19|20)\d{2})([.,!?;:]*)$', clean)
    if m_suit5:
        return f"{m_suit5.group(1).upper()}/{m_suit5.group(2).upper()}/{m_suit5.group(3).upper()}/{m_suit5.group(4).upper()}/{m_suit5.group(5)}{m_suit5.group(6)}"

    clean_stripped = lower.strip(PUNCT_CHARS)

    # Calendar months check
    if clean_stripped in CALENDAR_MONTHS_MAP:
        return apply_proper_case(clean, CALENDAR_MONTHS_MAP[clean_stripped])

    # Nigerian states check (single word)
    if clean_stripped in NIGERIAN_STATES_MAP:
        return apply_proper_case(clean, NIGERIAN_STATES_MAP[clean_stripped])

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
    clean_stripped = lower.strip(PUNCT_CHARS)
    if clean_stripped in COURT_TITLE_WORDS:
        proper = COURT_TITLE_WORDS[clean_stripped]
        return apply_proper_case(clean, proper)

    # Specific court acronyms
    court_acronyms = {"fhc", "nicn", "san", "sc", "ca"}
    if clean_stripped in court_acronyms:
        return apply_proper_case(clean, clean_stripped.upper())

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

        if i + 1 < n and is_number_or_word(curr_clean):
            next_item = words[i+1]
            next_text = str(next_item.get("word", "")).strip()
            next_clean = clean_token(next_text)

            is_valid_letter = False
            if len(next_clean) == 1 and next_clean.isalpha():
                if next_clean == "a":
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
    Merges sequences of word tokens separated by '/' into a single word token
    with NO space before or after the slash, e.g.:
    ['FHC', '/', 'ABJ', '/', 'CS', '/', '55', '/', '2024'] -> ['FHC/ABJ/CS/55/2024']
    ['plaintiff', '/', 'defendant'] -> ['plaintiff/defendant']
    ['one', '/', 'two'] -> ['1/2']
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

            # Format the merged slashed token (suit number uppercase vs normal words)
            parts = combined_text.split('/')
            formatted_parts = []
            has_digits = any(p.strip().isdigit() or p.strip().lower() in NUMBER_WORDS for p in parts)
            has_suit_acronyms = any(p.strip().lower() in ('fhc', 'nicn', 'ca', 'sc', 'cs', 'cr', 'abj', 'l', 'm', 'cv') for p in parts)
            is_suit_format = has_digits or has_suit_acronyms

            for p in parts:
                p_clean = p.strip()
                if p_clean.lower() in NUMBER_WORDS:
                    formatted_parts.append(str(NUMBER_WORDS[p_clean.lower()]))
                elif is_suit_format and (len(p_clean) <= 4 or p_clean.isupper()):
                    formatted_parts.append(p_clean.upper())
                else:
                    formatted_parts.append(p_clean)

            final_text = "/".join(formatted_parts)
            final_text = re.sub(r'/+', '/', final_text)

            merged.append({
                "word": final_text,
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
    """
    Normalizes a transcription segment:
    - Cleans up full text according to courtroom rules, slashes, numbers, and decorum.
    - Synchronizes individual word tokens, timestamps, and multi-word phrase casing.
    """
    if "text" in segment and segment["text"]:
        segment["text"] = normalize_case_numbers_and_slashes(segment["text"])

    if "words" in segment and segment["words"]:
        # 1. Clean individual word tokens
        for w in segment["words"]:
            if "word" in w and w["word"]:
                w["word"] = clean_word_token(w["word"])

        # 2. Merge number + letter combinations (e.g. ['5', 'a'] -> ['5A'])
        words = merge_number_letter_words(segment["words"])

        # 3. Merge slash sequences into unified tokens without spaces (e.g. ['plaintiff', '/', 'defendant'] -> ['plaintiff/defendant'])
        words = merge_slashed_words(words)

        # 4. Contextual honorific casing & court numbers for two-word/three-word sequences
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

        # 5. Exhibit casing synchronization across word tokens
        for idx in range(len(words)):
            w_clean = clean_token(words[idx].get("word", ""))
            if w_clean in ("exhibit", "exhibits"):
                has_num_follower = False
                if idx + 1 < len(words):
                    next_clean = clean_token(words[idx+1].get("word", ""))
                    if is_number_or_word(next_clean) or re.match(r'^[a-zA-Z0-9]+$', next_clean):
                        # Verify not a common non-identifier English word
                        if next_clean not in ("was", "is", "were", "are", "marked", "tendered", "admitted", "in", "to", "by", "that", "this"):
                            has_num_follower = True

                proper_ex = "Exhibits" if w_clean == "exhibits" else "Exhibit"
                if has_num_follower:
                    words[idx]["word"] = apply_proper_case(words[idx]["word"], proper_ex)
                else:
                    # Check if sentence start
                    is_start = (idx == 0) or any(words[idx-1]["word"].rstrip(PUNCT_CHARS).endswith(p) for p in ('.', '!', '?'))
                    if not is_start:
                        words[idx]["word"] = apply_proper_case(words[idx]["word"], w_clean.lower())

        # 6. Lowercase individual 'My' that is NOT at start of sentence and NOT part of an honorific
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

        segment["words"] = words

    return segment
