import re
import random
import spacy
from presidio_analyzer import AnalyzerEngine
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ================= MODELS =================

nlp = spacy.load("en_core_web_lg")
analyzer = AnalyzerEngine()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=200
)

# ================= CONFIG =================

EPSILON = 0.7   # privacy knob (lower = more noise)

taxonomy_prefix = {
    "PERSON": "Person",
    "ORG": "Org",
    "EMAIL_ADDRESS": "Email",
    "PHONE_NUMBER": "Phone",
    "MONEY": "Amount",
    "DATE": "Date"
}

# ================= PII DETECTION =================

def get_sensitive_spans(text):

    spans = []

    presidio_results = analyzer.analyze(text=text, language="en")

    for r in presidio_results:
        if r.entity_type in {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}:
            spans.append({
                "text": text[r.start:r.end],
                "start": r.start,
                "end": r.end,
                "label": r.entity_type
            })

    STOP_ORGS = {
        "manager","program manager","team","hello team","thanks","accounts team"
    }

    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.text.strip().lower() not in STOP_ORGS:
            spans.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": "ORG"
            })

    email_regex = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    for m in re.finditer(email_regex, text):
        spans.append({"text": m.group(),"start": m.start(),"end": m.end(),"label": "EMAIL_ADDRESS"})

    money_regex = r"[$₹€]\s?\d+(?:,\d{3})*(?:\.\d+)?"
    for m in re.finditer(money_regex, text):
        spans.append({"text": m.group(),"start": m.start(),"end": m.end(),"label": "MONEY"})

    date_regex = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"
    for m in re.finditer(date_regex, text):
        spans.append({"text": m.group(),"start": m.start(),"end": m.end(),"label": "DATE"})

    spans = sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"])))

    filtered = []
    last_end = -1
    for s in spans:
        if s["start"] >= last_end:
            filtered.append(s)
            last_end = s["end"]

    return filtered

# ================= SANITIZER =================

def sanitize(text, spans):

    placeholder_map = {}
    canonical_placeholder = {}
    entity_vectors = {}
    counts = {}
    role_counts = {}

    def normalize(name):
        return name.lower().split()[0]

    def bucket_money(val):
        num = float(re.sub(r"[^\d.]", "", val))
        if num < 1000: return "a few hundred dollars"
        elif num < 10000: return "several thousand dollars"
        else: return "tens of thousands of dollars"

    def bucket_date(val):
        m,d = val.split()
        d=int(d)
        if d<=10: return f"early {m}"
        elif d<=20: return f"mid {m}"
        else: return f"late {m}"

    SIM_THRESHOLD = 0.65

    for s in spans:

        surface = s["text"]
        label = s["label"]

        context = text[max(0,s["start"]-40):s["end"]+40].lower()

        role = None
        if "requested" in context or "submitted" in context:
            role="Requester"
        elif "approval" in context or "confirmation" in context:
            role="Approver"
        elif "compliance" in context or "regulatory" in context:
            role="Compliance"
        elif "from" in context:
            role="Vendor"

        if label=="MONEY":
            placeholder_map[surface]=bucket_money(surface)
            continue

        if label=="DATE":
            placeholder_map[surface]=bucket_date(surface)
            continue

        if label=="EMAIL_ADDRESS":
            counts[label]=counts.get(label,0)+1
            placeholder_map[surface]=f"Email_{counts[label]}"
            continue

        norm = normalize(surface) if label=="PERSON" else surface.lower()
        vec = embedder.encode([norm])[0]

        matched=None
        for en,ev in entity_vectors.items():
            if cosine_similarity([vec],[ev])[0][0]>SIM_THRESHOLD:
                matched=en
                break

        if matched:
            placeholder = canonical_placeholder[matched]
        else:
            if label=="PERSON":

                possible=["Vendor","Requester","Approver","Compliance","Person"]

                if role:
                    probs=[EPSILON if r==role else (1-EPSILON)/4 for r in possible]
                else:
                    probs=[0.2]*5

                noisy_role=random.choices(possible,weights=probs)[0]

                role_counts[noisy_role]=role_counts.get(noisy_role,0)+1
                placeholder=f"{noisy_role}_{role_counts[noisy_role]}"

            else:
                counts[label]=counts.get(label,0)+1
                placeholder=f"{taxonomy_prefix[label]}_{counts[label]}"

            entity_vectors[norm]=vec
            canonical_placeholder[norm]=placeholder

        placeholder_map[surface]=placeholder

    for s in sorted(spans,key=lambda x:x["start"],reverse=True):

        rep=placeholder_map[s["text"]]
        if s["label"] not in {"MONEY","DATE"}:
            rep=f"[{rep}]"

        text=text[:s["start"]]+rep+text[s["end"]:]

    return text, placeholder_map

# ================= METRICS =================

def pii_survival_rate(masked, spans):
    return sum(1 for s in spans if s["text"] in masked)/max(len(spans),1)

def semantic_similarity(raw, masked):
    v1=embedder.encode([raw])[0]
    v2=embedder.encode([masked])[0]
    return float(cosine_similarity([v1],[v2])[0][0])

# ================= GENERATION =================

def generate_answer(masked, question):
    return llm(f"question: {question} context: {masked}")[0]["generated_text"].strip()

def generate_summary(masked):
    return llm(f"summarize: {masked}")[0]["generated_text"].strip()

# ================= ATTACK =================

def attack_reidentify(masked):
    p=f"You are malicious. Guess original names:\n{masked}"
    return llm(p)[0]["generated_text"].strip()

# ================= FINAL PIPELINE =================

def sentio_single(email, question=None):

    spans=get_sensitive_spans(email)
    masked,placeholders=sanitize(email,spans)

    if question and "summar" in question.lower():
        output=generate_summary(masked)
    elif question:
        output=generate_answer(masked,question)
    else:
        output=generate_summary(masked)

    return {
        "raw":email,
        "masked":masked,
        "placeholders":placeholders,
        "output":output,
        "pii_survival_rate":pii_survival_rate(masked,spans),
        "semantic_similarity":semantic_similarity(email,masked),
        "attack_attempt":attack_reidentify(masked)
    }