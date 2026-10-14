"""
Medical Chatbot — FastAPI Backend
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000

Folder structure expected:
mini project/
├── main.py
├── kb_embeddings.npy
├── medquad_faiss.index
├── kb_metadata.pkl
├── label_encoder.pkl
├── intent_classes.json
└── intent_model_final/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
"""
current_report_text = ""
current_report_id = None
from fastapi import UploadFile, File
import os, json, pickle, re, time
import numpy as np
import faiss
import torch
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session
from fastapi import Depends

from database import get_db
from db_models import User
from schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    ReportQuestion
)
from auth import (
    hash_password,
    verify_password,
    create_access_token
)
from fastapi import Depends, Header
from auth import decode_access_token
from db_models import User
from fastapi import Depends, Header
from jose import JWTError

from auth import decode_access_token
from db_models import User, Conversation, Message
from sqlalchemy import or_
from reports import (
    extract_pdf_text,
    summarize_report
)
from db_models import MedicalReport
# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — all files must be in the SAME folder as main.py
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

KB_EMBEDDINGS = os.path.join(DATA_DIR, "kb_embeddings.npy")
FAISS_INDEX = os.path.join(DATA_DIR, "medquad_faiss.index")
KB_METADATA = os.path.join(DATA_DIR, "kb_metadata.pkl")

LABEL_ENCODER = os.path.join(MODELS_DIR, "label_encoder.pkl")
INTENT_CLASSES = os.path.join(MODELS_DIR, "intent_classes.json")

INTENT_MODEL = MODELS_DIR
# ─────────────────────────────────────────────────────────────────────────────
# STARTUP — load everything once
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Medical Chatbot API — Loading artifacts...")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# 1. Knowledge Base metadata
print("Loading KB metadata...")
with open(KB_METADATA, "rb") as f:
    kb = pickle.load(f)
kb_questions = kb["questions"]
kb_answers   = kb["answers"]
kb_intents   = kb["intents"]
print(f"  KB size: {len(kb_questions)} QA pairs")

# 2. FAISS index + embeddings
print("Loading FAISS index...")
kb_embeddings = np.load(KB_EMBEDDINGS).astype("float32")
index         = faiss.read_index(FAISS_INDEX)
print(f"  FAISS vectors: {index.ntotal} | dim: {kb_embeddings.shape[1]}")

# 3. Label encoder
print("Loading label encoder...")
with open(LABEL_ENCODER, "rb") as f:
    label_encoder = pickle.load(f)

# 4. Intent classes
print("Loading intent classes...")
with open(INTENT_CLASSES) as f:
    raw = json.load(f)
# handle both {0: "class"} and ["class", ...] formats
if isinstance(raw, dict):
    intent_classes = [raw[str(i)] for i in range(len(raw))]
else:
    intent_classes = raw
print(f"  Intents: {intent_classes}")

# 5. Sentence Transformer (embedding model)
print("Loading MiniLM embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()
print(f"  Embedding dim: {EMBED_DIM}")

# 6. BioBERT Intent Classifier
print("Loading BioBERT intent classifier...")
intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL)
intent_model_clf = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL)
intent_model_clf = intent_model_clf.to(device)
intent_model_clf.eval()
print(f"  Model loaded from: {INTENT_MODEL}")

# 7. spaCy
print("Loading spaCy...")
nlp_spacy = spacy.load("en_core_web_sm")

print("="*60)
print("  ✅ All artifacts loaded. API ready.")
print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (exact copies from your notebook)
# ─────────────────────────────────────────────────────────────────────────────

def predict_intent(question: str):
    """Predict intent — rule overrides first, then BioBERT."""
    q = question.lower()

    # Rule overrides (same as notebook)
    if re.search(r'\b(treat|treatment|therapy|medicine|medication|drug|surgery)\b', q):
        return "treatment", 1.0
    if re.search(r'\b(cause|causes|why|reason|etiology)\b', q):
        return "causes", 1.0
    if re.search(r'\b(symptom|symptoms|sign|signs)\b', q):
        return "symptoms", 1.0
    if re.search(r'\b(risk factor|at risk|who gets|susceptible)\b', q):
        return "risk_factors", 1.0
    if re.search(r'\b(prevent|prevention|avoid|reduce risk)\b', q):
        return "prevention", 1.0
    if re.search(r'\b(diagnos|test|testing|detect|screen)\b', q):
        return "diagnosis", 1.0
    if re.search(r'\b(prognosis|outlook|survival|recovery)\b', q):
        return "prognosis", 1.0
    if re.search(r'\b(what is|what are|define|meaning of)\b', q):
        return "definition", 1.0

    # BioBERT fallback
    inputs = intent_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    ).to(device)

    with torch.no_grad():
        logits = intent_model_clf(**inputs).logits

    pred_id  = torch.argmax(logits, dim=-1).item()
    pred_prob = torch.softmax(logits, dim=-1).max().item()
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent, float(pred_prob)


def extract_entities(text: str):
    """Entity extraction — exact copy from notebook Cell 12."""
    text = text.lower().replace("'s", "")
    doc  = nlp_spacy(text)
    entities = []

    # regex for strong medical patterns
    pattern = r'(type\s*\d+\s+[a-z]+)'
    matches = re.findall(pattern, text)
    entities.extend(matches)

    stopwords = {
        'what','who','how','when','where','which','why','the',
        'is','are','was','were','be','been','being',
        'have','has','had','do','does','did',
        'will','would','can','could','may','might',
        'symptom','symptoms','cause','causes','treatment','treatments',
        'sign','signs','effect','effects','risk','risks',
        'people','person','patient','doctor',
        'information','question','answer',
        'condition','disease'
    }

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        words      = chunk_text.split()
        filtered   = [w for w in words if w not in stopwords]
        if not filtered:
            continue
        phrase = " ".join(filtered)
        if 1 <= len(filtered) <= 4:
            entities.append(phrase)

    # special fixes
    fixed = []
    for e in entities:
        e = e.strip()
        if "alzheimer" in e:
            fixed.append("alzheimers disease")
        else:
            fixed.append(e)

    # keep longest phrases
    final = []
    for e in fixed:
        if not any(e != other and e in other for other in fixed):
            final.append(e)

    # clean + deduplicate
    clean = []
    seen  = set()
    for e in final:
        e = re.sub(r'[^a-z0-9 ]', '', e).strip()
        if e and e not in seen and len(e) > 2:
            seen.add(e)
            clean.append(e)

    return clean


def entity_match(ent: str, text: str) -> bool:
    """Strict entity match — exact copy from notebook Cell 13."""
    ent  = ent.lower().strip()
    text = text.lower()
    if ent in text:
        return True
    words = ent.split()
    if len(words) > 1 and all(w in text for w in words):
        return True
    return False


def smart_retrieve(query: str, top_k: int = 5, min_score: float = 0.30):
    """Full pipeline — exact copy from notebook Cell 13."""
    # 1. Intent
    intent, intent_conf = predict_intent(query)
    if intent_conf < 0.90:
        intent = "general"

    # 2. Entities
    entities = extract_entities(query)

    # 3. FAISS search (3x candidates)
    t0 = time.perf_counter()
    query_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(query_emb, top_k * 3)
    search_ms = (time.perf_counter() - t0) * 1000

    # 4. Build candidates with boosts
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        base  = float(score)
        kb_q  = kb_questions[idx].lower()
        kb_a  = kb_answers[idx].lower()

        ib = 0.25 if kb_intents[idx] == intent else 0.0
        eb = 0.0
        for ent in entities:
            if entity_match(ent, kb_q) or entity_match(ent, kb_a):
                eb = 0.25
                break

        candidates.append({
            "question"    : kb_questions[idx],
            "answer"      : kb_answers[idx],
            "base_score"  : base,
            "intent_boost": ib,
            "entity_boost": eb,
            "final_score" : base + ib + eb,
            "intent"      : kb_intents[idx],
        })

    # 5. Strict entity filter
    entity_filtered = []
    for c in candidates:
        for ent in entities:
            if entity_match(ent, c["question"]) or entity_match(ent, c["answer"]):
                entity_filtered.append(c)
                break

    if entities and not entity_filtered:
        return [], intent, intent_conf, entities, search_ms

    if not entities:
        entity_filtered = candidates

    # 6. Intent filter
    intent_filtered = [c for c in entity_filtered if c["intent"] == intent]
    if not intent_filtered:
        intent_filtered = entity_filtered

    # 7. Sort + threshold
    intent_filtered.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = [c for c in intent_filtered[:top_k] if c["base_score"] >= min_score]

    return top_results, intent, intent_conf, entities, search_ms

def format_as_bullets(text):
    if not text:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 2:
        return text

    intro = sentences[0]

    bullets = []
    for sentence in sentences[1:]:
        bullets.append(f"• {sentence}")

    return intro + "\n\n" + "\n".join(bullets)

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Chatbot API",
    description="Semantic retrieval chatbot using BioBERT + MiniLM + FAISS",
    version="1.0.0"
)

# CORS — allow React frontend on any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    conversation_id: int | None = None

class ConversationResponse(BaseModel):
    id: int
    title: str

class TopResult(BaseModel):
    question: str
    answer:   str
    score:    float


class ChatResponse(BaseModel):
    answer: str
    intent: str
    intent_conf: float
    entities: list
    matched_q: str
    score: float
    response_ms: float
    top_results: list

    conversation_id: int | None = None

def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing token"
        )

    token = authorization.replace(
        "Bearer ",
        ""
    )

    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    user = db.query(User).filter(
        User.id == payload["user_id"]
    ).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status"  : "Medical Chatbot API is running",
        "kb_size" : len(kb_questions),
        "intents" : intent_classes,
        "docs"    : "/docs"
    }


@app.get("/health")
def health():
    return {
        "status"     : "ok",
        "kb_size"    : len(kb_questions),
        "faiss_vecs" : int(index.ntotal),
        "device"     : device,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = req.question.strip()
    print("QUESTION =", q)
    print("WORD COUNT =", len(q.split()))
    conversation_id = req.conversation_id

    if conversation_id is None:

        conversation = Conversation(
            user_id=current_user.id,
            title=q[:100]
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        conversation_id = conversation.id

    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(q) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars).")
    if len(q.split()) < 3:
        return ChatResponse(
            answer="Please ask a more specific medical question.",
            intent="unknown",
            intent_conf=0.0,
            entities=[],
            matched_q="",
            score=0.0,
            response_ms=0,
            top_results=[],
            conversation_id=conversation_id
        )

    t0 = time.perf_counter()
    results, intent, conf, entities, search_ms = smart_retrieve(
        q, top_k=req.top_k
    )

    response_ms = (time.perf_counter() - t0) * 1000
    user_message = Message(
        conversation_id=conversation_id,
        sender="user",
        message=q
    )
    db.add(user_message)
    db.commit()

    if not results:
        return ChatResponse(
            answer="Please ask a more specific medical question.",
            intent="unknown",
            intent_conf=0.0,
            entities=[],
            matched_q="",
            score=0.0,
            response_ms=0,
            top_results=[],
            conversation_id=conversation_id
        )

    best = results[0]

    # Truncate very long answers for clean display
    answer_text = best["answer"]
    if len(answer_text.split()) > 250:
        answer_text = " ".join(answer_text.split()[:250]) + "..."

    answer_text = format_as_bullets(answer_text)
    bot_message = Message(
        conversation_id=conversation_id,
        sender="bot",
        message=answer_text
    )

    db.add(bot_message)
    db.commit()
    return ChatResponse(
            answer=answer_text,
            intent=intent,
            intent_conf=conf,
            entities=entities,
            matched_q=best["question"],
            score=round(best["final_score"], 4),
            response_ms=round(response_ms, 1),
            top_results=[
                {
                    "question": r["question"],
                    "answer": r["answer"][:200] + "..."
                    if len(r["answer"]) > 200
                    else r["answer"],
                    "score": round(r["final_score"], 4)
                }
                for r in results
            ],
            conversation_id=conversation_id
        )


@app.get("/intents")
def get_intents():
    return {"intents": intent_classes}


# ─────────────────────────────────────────────────────────────────────────────
# RUN (only when executing directly, not via uvicorn)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

@app.post("/register")
def register_user(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):
    existing_email = db.query(User).filter(
        User.email == user_data.email
    ).first()

    if existing_email:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    existing_username = db.query(User).filter(
        User.username == user_data.username
    ).first()

    if existing_username:
        raise HTTPException(
            status_code=400,
            detail="Username already taken"
        )

    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hash_password(
            user_data.password
        )
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully"
    }

@app.post("/login", response_model=TokenResponse)
def login_user(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
    or_(
        User.email == login_data.login,
        User.username == login_data.login
    )
).first()
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    if not verify_password(
        login_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(
        {
            "sub": user.email,
            "user_id": user.id
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }
@app.get("/me")
def get_me(
    current_user: User = Depends(get_current_user)
):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }
@app.post("/conversations")
def create_conversation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversation = Conversation(
        user_id=current_user.id,
        title="New Chat"
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "id": conversation.id,
        "title": conversation.title
    }
@app.get("/conversations")
def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .all()
    )

    return conversations

@app.get("/messages/{conversation_id}")
def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at.asc())
        .all()
    )

    return messages
@app.post("/upload-report")
async def upload_report(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            await file.read()
        )

    report_text = extract_pdf_text(
    file_path
    )
    report = MedicalReport(
    user_id=current_user.id,    # temporarily hardcode your user id
    filename=file.filename,
    report_text=report_text
)

    db.add(report)

    db.commit()
    global current_report_text

    current_report_text = report_text
    summary = summarize_report(
        report_text
    )

    return {
        "filename": file.filename,
        "summary": summary
    }
def extract_test_values(report_text):

    lines = [
        line.strip()
        for line in report_text.split("\n")
        if line.strip()
    ]

    results = {}

    bad_words = [

        "verified",
        "consultant",
        "permission",
        "report",
        "suggested",
        "clinical",
        "laboratory",
        "tested",
        "resident",
        "pathologist",
        "biochemist",
        "mc-",
        "end of report",
        "received dt",
        "reported dt",
        "date",
        "time"

    ]

    for i in range(len(lines)-1):

        current_line = lines[i]
        next_line = lines[i+1]

        if not re.search(r"\d", next_line):
            continue

        if any(
            word in current_line.lower()
            for word in bad_words
        ):
            continue

        if len(current_line) < 3:
            continue

        results[current_line.lower()] = next_line

    return results
@app.post("/report-chat")
def report_chat(req: ReportQuestion):

    global current_report_text

    if not current_report_text:

        raise HTTPException(
            status_code=400,
            detail="No report uploaded."
        )

    tests = extract_test_values(
        current_report_text
    )

    question = req.question.lower()

    matches = []
    if "summary" in question or "summarize" in question:

            summary = []

            for test_name, value in tests.items():

                summary.append(
                    f"• {test_name.title()} : {value}"
                )

                if len(summary) == 15:
                    break

            return {

                "answer":
                "📄 Report Summary\n\n" +
                "\n".join(summary)

            }
        
    for test_name, value in tests.items():

        if any(
            word in test_name
            for word in question.split()
        ):

            matches.append(
                f"• {test_name.title()} : {value}"
            )

    if not matches:

        return {

            "answer":
            "I could not find relevant information in the uploaded report."

        }

    return {

        "answer":
        "🔬 Results found:\n\n" +
        "\n".join(matches)

    }
@app.get("/reports")
def get_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    reports = (
        db.query(MedicalReport)
        .filter(
            MedicalReport.user_id == current_user.id
        )
        .order_by(
            MedicalReport.created_at.desc()
        )
        .all()
    )

    return reports
@app.get("/report/{report_id}")
def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    report = (
        db.query(MedicalReport)
        .filter(
            MedicalReport.id == report_id,
            MedicalReport.user_id == current_user.id
        )
        .first()
    )

    if not report:

        raise HTTPException(
            status_code=404,
            detail="Report not found."
        )
    global current_report_text

    current_report_text = report.report_text
    return {
        "message":
            f"{report.filename} activated."
    }