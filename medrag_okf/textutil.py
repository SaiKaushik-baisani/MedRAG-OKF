"""Small text helpers shared by retrieval, verification and query analysis."""
from __future__ import annotations

import re

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with without from into onto over under about as is are was were be been
    being am do does did doing have has had having it its this that these those i me my we our you your he she they them their what
    which who whom whose when where why how can could should would may might will shall not no nor so than too very just also more
    most some any each other such only own same there here up down out off again further once all both few many much s t don now
    get got gets getting use used using like need needs make makes take taken takes please tell explain give want know often
    usually sometimes people person someone one two always never every even after before still really ever already""".split()
)

# Words that appear in almost every medical question and carry no topical signal.
GENERIC_MEDICAL = set(
    """treatment treat treated treatments cause causes caused symptom symptoms sign signs help helps prevent prevention preventing
    manage managing management long term short risk risks factor factors good bad common best way ways medical medicine health
    healthy condition disease problem problems doctor see visit normal feel feeling feels differ different difference better night time level levels fast quickly effect effects side""".split()
)

# Light query expansion so the lexical retriever bridges everyday words and clinical words.
SYNONYMS = {
    "tired": ["fatigue", "tiredness", "exhaustion"],
    "tiredness": ["fatigue"],
    "exhausted": ["fatigue"],
    "sugar": ["glucose", "diabetes"],
    "bp": ["blood", "pressure", "hypertension"],
    "pressure": ["hypertension"],
    "heartburn": ["reflux", "gerd"],
    "acidity": ["reflux", "acid"],
    "pee": ["urine", "urinary"],
    "peeing": ["urine", "urinary"],
    "urination": ["urine", "urinary"],
    "burning": ["uti", "urinary"],
    "flu": ["influenza"],
    "cold": ["common", "cold", "rhinovirus"],
    "wheezing": ["asthma"],
    "inhaler": ["asthma"],
    "thyroid": ["hypothyroidism"],
    "anaemia": ["anemia", "iron"],
    "anemia": ["iron", "haemoglobin"],
    "hemoglobin": ["anemia", "iron"],
    "haemoglobin": ["anemia", "iron"],
    "sunlight": ["vitamin", "d"],
    "mosquito": ["dengue"],
    "platelet": ["dengue"],
    "platelets": ["dengue"],
    "headache": ["migraine", "headache"],
    "insomnia": ["sleep"],
    "sleepless": ["sleep", "insomnia"],
    "dehydrated": ["dehydration", "fluids"],
    "thirsty": ["dehydration", "thirst"],
    "drink": ["fluids", "water"],
}

_SUFFIXES = ("ations", "ation", "ingly", "ings", "ment", "ness", "ing", "edly", "ies", "ied", "es", "ed", "ly", "s")


def stem(token: str) -> str:
    """Very small suffix stripper (good enough for BM25 on short medical text).
    Applied up to twice (tiredness -> tired -> tir) and a trailing 'e' is dropped so migraine/migraines match."""
    for _ in range(2):
        if len(token) <= 3:
            break
        for suf in _SUFFIXES:
            if token.endswith(suf) and len(token) - len(suf) >= 3:
                base = token[: -len(suf)]
                token = base + "y" if suf in ("ies", "ied") else base
                break
        else:
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?")


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if not keep_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


def content_terms(text: str) -> list[str]:
    """Topical terms only (stop-words and generic medical filler removed)."""
    raw = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and t not in GENERIC_MEDICAL]
    return [stem(t) for t in raw]


def expand_query(text: str) -> list[str]:
    """Tokens for the query plus a few clinical synonyms."""
    base = _TOKEN_RE.findall(text.lower())
    extra: list[str] = []
    for t in base:
        extra.extend(SYNONYMS.get(t, []))
    return [stem(t) for t in base + extra if t not in STOPWORDS]


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return [re.sub(r"^[-*•]\s*", "", p) for p in parts]


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def content_groups(text: str) -> list[set[str]]:
    """One group per topical query word: the word's stem plus its clinical synonyms (a hit on any member counts)."""
    groups, seen = [], set()
    for t in _TOKEN_RE.findall(text.lower()):
        if t in STOPWORDS or t in GENERIC_MEDICAL:
            continue
        st = stem(t)
        if st in seen:
            continue
        seen.add(st)
        groups.append({st} | {stem(x) for x in SYNONYMS.get(t, [])})
    return groups
