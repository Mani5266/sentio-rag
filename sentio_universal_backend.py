import re
from presidio_analyzer import AnalyzerEngine
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= MODELS =================

analyzer = AnalyzerEngine()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# ================= CONFIG =================

taxonomy_prefix = {
    "PER": "Person",
    "ORG": "Org",
    "EMAIL": "Email",
    "PHONE": "Phone",
    "MONEY": "Amount",
    "DATE": "Date",
    "ID": "StudentID"
}

SIM_THRESHOLD = 0.65

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_REGEX = r"(?:\+91\s*)?\d{10}"
ROLL_REGEX = r"\b\d{4,}[A-Z]\d{3,}\b"
MONEY_REGEX = r"(?:₹|\$|€)\s?\d[\d,]*"
DATE_REGEX = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"

# Indian-style full names (fallback)
NAME_REGEX = r"\b(?:[A-Z]\s)?[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}\b"

# ================= HELPERS =================

def bucket_money(val):
    num = float(re.sub(r"[^\d]", "", val))
    if num < 1000:
        return "hundreds"
    elif num < 10000:
        return "thousands"
    else:
        return "tens of thousands"

def bucket_date(val):
    m, d = val.split()
    d = int(d)
    if d <= 10:
        return f"early {m}"
    elif d <= 20:
        return f"mid {m}"
    else:
        return f"late {m}"

def normalize(x):
    return x.lower().strip()

def normalize_initials(text):
    return re.sub(r"\b([A-Z])\.", r"\1 ", text)

# ================= SPAN DETECTOR =================

def get_spans(text):

    text = normalize_initials(text)
    spans = []

    # Transformer NER
    ents = ner(text)
    for e in ents:
        w = e["word"].strip()
        if len(w) < 3 or "##" in w:
            continue
        if w.isupper() and len(w.split()) > 3:
            continue
        if e["entity_group"] in {"PER","ORG"}:
            spans.append({"text": w, "label": e["entity_group"]})

    # Fallback name regex (Indian names)
    for m in re.finditer(NAME_REGEX, text):
        spans.append({"text": m.group(), "label": "PER"})

    # Email
    for m in re.finditer(EMAIL_REGEX, text):
        spans.append({"text": m.group(), "label": "EMAIL"})

    phones=[]
    for m in re.finditer(PHONE_REGEX, text):
        phones.append(m.group())
        spans.append({"text": m.group(), "label": "PHONE"})

    for m in re.finditer(MONEY_REGEX, text):
        spans.append({"text": m.group(), "label": "MONEY"})

    for m in re.finditer(DATE_REGEX, text):
        spans.append({"text": m.group(), "label": "DATE"})

    for m in re.finditer(ROLL_REGEX, text):
        if m.group() not in phones:
            spans.append({"text": m.group(), "label": "ID"})

    return spans

# ================= SANITIZER =================

def sanitize_with_mapping(text):

    text = normalize_initials(text)
    spans = get_spans(text)

    placeholder={}
    canon={}
    vectors={}
    counts={}

    for s in spans:

        surface=s["text"]
        label=s["label"]

        if surface in placeholder:
            continue

        if label in {"EMAIL","PHONE","ID"}:
            counts[label]=counts.get(label,0)+1
            placeholder[surface]=f"{taxonomy_prefix[label]}_{counts[label]}"
            continue

        if label=="MONEY":
            placeholder[surface]=bucket_money(surface)
            continue

        if label=="DATE":
            placeholder[surface]=bucket_date(surface)
            continue

        norm=normalize(surface)
        vec=embedder.encode([norm])[0]

        match=None
        for k,v in vectors.items():
            if cosine_similarity([vec],[v])[0][0]>SIM_THRESHOLD:
                match=k
                break

        if match:
            ph=canon[match]
        else:
            counts[label]=counts.get(label,0)+1
            ph=f"{taxonomy_prefix[label]}_{counts[label]}"
            vectors[norm]=vec
            canon[norm]=ph

        placeholder[surface]=ph

    # Money + date direct
    for k,v in placeholder.items():
        if v in {"hundreds","thousands","tens of thousands"} or v.startswith(("early","mid","late")):
            text=text.replace(k,v)

    # Safe regex replace
    for k,v in placeholder.items():
        if v in {"hundreds","thousands","tens of thousands"} or v.startswith(("early","mid","late")):
            continue
        text=re.sub(rf"\b{re.escape(k)}\b",f"[{v}]",text)

    return text, placeholder

# ================= API =================

def sentio_universal(text):
    return sanitize_with_mapping(text)