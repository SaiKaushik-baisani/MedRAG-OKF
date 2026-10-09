"""Query analysis: safety / triage layer (predefined high-risk patterns) and query clarification."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textutil import content_terms

DISCLAIMER = "MedRAG_OKF is an assistive information system, not a replacement for a medical professional."

EMERGENCY_FOOTER = (
    "Call your local emergency number now (112 in India) or go to the nearest emergency department. "
    "Do not wait for an online answer."
)


@dataclass
class Triage:
    level: str = "none"            # none | caution | emergency
    category: str = ""
    message: str = ""
    matched: list[str] = field(default_factory=list)


# (category, regex list, message)
_EMERGENCY_RULES: list[tuple[str, list[str], str]] = [
    (
        "Possible heart attack",
        [r"chest (pain|pressure|tightness)", r"pain.{0,30}(left arm|jaw)", r"heart attack"],
        "Chest pain or pressure, especially with sweating, breathlessness or pain spreading to the arm or jaw, can be a heart emergency.",
    ),
    (
        "Possible stroke",
        [r"face droop", r"slurred speech", r"sudden(ly)? (weakness|numbness).{0,25}(one side|arm|leg|face)", r"\bstroke\b"],
        "Face drooping, arm weakness or slurred speech can be signs of a stroke, where minutes matter.",
    ),
    (
        "Severe breathing difficulty",
        [r"(can'?t|cannot|unable to|struggling to|trouble|difficulty) breath", r"gasping for (air|breath)", r"lips? (are |is )?(turning )?blue"],
        "Severe difficulty breathing is an emergency.",
    ),
    (
        "Suicidal thoughts / self-harm",
        [r"(kill|hurt|harm) myself", r"want to die", r"suicid", r"end my life", r"don'?t want to live"],
        "You are not alone, and help is available right now. In India you can call Tele-MANAS at 14416 (free, 24x7), or reach emergency services on 112. "
        "If you can, tell someone you trust where you are and stay with them.",
    ),
    (
        "Severe allergic reaction",
        [r"anaphyla", r"throat (is )?(closing|swelling)", r"swelling of (the )?(face|tongue|lips).{0,30}(breath|swallow)"],
        "A severe allergic reaction can progress within minutes.",
    ),
    (
        "Poisoning / overdose",
        [r"overdos", r"swallowed (poison|pesticide|bleach)", r"took too many (pills|tablets)", r"poison"],
        "Possible poisoning or overdose needs urgent professional help.",
    ),
    (
        "Seizure / unconsciousness",
        [r"\bseizure", r"\bconvuls", r"unconscious", r"not responding", r"passed out and"],
        "A seizure that is new, lasts more than a few minutes, or an unresponsive person is an emergency.",
    ),
    (
        "Heavy bleeding",
        [r"(heavy|severe|uncontrolled) bleeding", r"won'?t stop bleeding", r"vomiting blood", r"coughing (up )?blood"],
        "Heavy or uncontrolled bleeding is an emergency.",
    ),
    (
        "Fever in a young infant",
        [r"(newborn|baby|infant).{0,40}(fever|high temperature)", r"(fever|high temperature).{0,40}(newborn|baby|infant)"],
        "Fever in a young baby needs prompt medical assessment.",
    ),
]

_CAUTION_RULES: list[tuple[str, list[str], str]] = [
    (
        "Medication dosing",
        [r"how (much|many).{0,40}(mg|tablet|pill|dose|paracetamol|ibuprofen|medicine)", r"\bdosage\b", r"\bdose\b.{0,20}(of|for)", r"how much .{0,25}should i take"],
        "I can't recommend personal medicine doses. Dosing depends on age, weight, other conditions and the exact product. Please follow the label or ask a pharmacist or doctor.",
    ),
    (
        "Diagnosis request",
        [r"do i have", r"am i (suffering|dying)", r"diagnose me", r"what('?s| is) wrong with me", r"is it (cancer|serious)"],
        "I can share general information but I can't diagnose you. A clinician needs to examine you and may need tests.",
    ),
    (
        "Stopping prescribed treatment",
        [r"(stop|quit|skip).{0,20}(my )?(medicine|medication|insulin|tablets|pills)"],
        "Please do not stop or change prescribed medicine without talking to the doctor who prescribed it.",
    ),
]


def assess_safety(query: str) -> Triage:
    q = query.lower()
    for category, patterns, msg in _EMERGENCY_RULES:
        hits = [p for p in patterns if re.search(p, q)]
        if hits:
            return Triage("emergency", category, msg + " " + EMERGENCY_FOOTER if category != "Suicidal thoughts / self-harm" else msg, hits)
    for category, patterns, msg in _CAUTION_RULES:
        hits = [p for p in patterns if re.search(p, q)]
        if hits:
            return Triage("caution", category, msg, hits)
    return Triage()


# ------------------------------------------------------------------ clarification
_VAGUE_TOPICS: list[tuple[list[str], str, list[str]]] = [
    (["pain", "ache", "hurts", "hurt"], "pain", [
        "Where exactly is the pain (head, chest, stomach, back, joints...)?",
        "How long has it been going on, and is it getting worse?",
        "Is anything else happening with it (fever, vomiting, breathlessness)?",
    ]),
    (["sick", "unwell", "ill", "bad"], "feeling unwell", [
        "What symptoms are you noticing (fever, cough, tiredness, stomach upset...)?",
        "When did they start?",
    ]),
    (["medicine", "medication", "tablet", "tablets", "pill", "pills", "drug"], "medicine", [
        "Which medicine or condition are you asking about?",
        "Are you asking about how it works, side effects, or when to take it?",
    ]),
    (["problem", "issue", "thing", "something"], "unspecified problem", [
        "Which symptom or condition would you like information on?",
        "How long has it been present?",
    ]),
]

_KNOWN_TOPIC_HINTS = set(
    """fatigue tired tiredness diabetes sugar glucose hypertension pressure bp anemia anaemia iron asthma wheeze cold flu influenza
    migraine headache dehydration dehydrated uti urinary urine vitamin thyroid hypothyroidism reflux gerd heartburn dengue sleep insomnia
    fever cough""".split()
)


def needs_clarification(query: str) -> tuple[bool, list[str]]:
    """Ask focused questions when a query is too vague to retrieve against."""
    q = query.lower()
    terms = content_terms(query)
    raw = set(re.findall(r"[a-z]+", q))
    if raw & _KNOWN_TOPIC_HINTS:
        return False, []
    for keywords, _, questions in _VAGUE_TOPICS:
        if raw & set(keywords) and len(terms) <= 3:
            return True, questions
    if len(terms) == 0:
        return True, [
            "Could you describe the symptom or condition in a bit more detail?",
            "How long has it been going on?",
        ]
    return False, []
