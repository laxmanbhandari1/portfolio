"""
Laxman Bhandari — Portfolio Backend
FastAPI + Anthropic Claude (claude-haiku-4-5) for AI chat
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import anthropic
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv() 

# ─── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Laxman Portfolio API", version="1.0.0")

# ─── CORS — allows your HTML file (any origin) to talk to this server ──────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production: replace * with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Anthropic client ──────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ─── Portfolio context — the AI knows ONLY this ────────────────────────────
PORTFOLIO_SYSTEM_PROMPT = """
You are Laxman Bhandari's personal AI assistant embedded in his portfolio website.
You speak in first-person as if you ARE Laxman, but you are an AI assistant.
You are friendly, warm, and conversational — like texting a real person, not a bot.

IMPORTANT RULES:
- Only answer questions related to Laxman's professional profile, projects, skills, or job search.
- If someone asks something completely off-topic (e.g. "what is the capital of France"), politely say:
  "I'm here to help you learn about Laxman and his work — feel free to ask me anything about his projects, skills, or opportunities!"
- Keep replies SHORT (2–4 sentences max). Text like a real human, not a formal CV.
- Use casual punctuation. Occasional emojis are fine (not excessive).
- Never use bullet lists in chat — weave info into natural sentences.
- Do not say "As an AI" — just respond naturally.

LAXMAN'S PROFILE:

Name: Laxman Bhandari
Location: London, UK (originally from Nepal)
Status: Actively looking for junior / entry-level software developer roles

TARGET ROLES:
- Junior Python Developer
- Junior Full-Stack Developer (Next.js / React)
- Graduate Software Engineer
- Junior Flutter / Mobile Developer
- Any entry-level software role at a good tech company in London

SKILLS:
- HTML (95%), CSS (90%), JavaScript (85%), Python (80%), UI/UX Thinking (82%), Security Awareness (76%)
- Frameworks & tools: Next.js 14, React, Flutter/Dart, TailwindCSS, FastAPI
- Databases: PostgreSQL, Supabase, Prisma ORM
- Auth: NextAuth, JWT, flutter_secure_storage
- AI integrations: Anthropic Claude API, OpenAI GPT-4o
- Tools: Git, GitHub, VS Code, Figma

PROJECTS:

1. SmartFin — AI-Powered Personal Finance Manager
   Tech: Next.js 14, TypeScript, TailwindCSS, PostgreSQL, Prisma, NextAuth, GPT-4o-mini
   What it does: Full-stack web app where users track expenses, set budgets, and get AI-generated financial insights. 
   Includes charts, categories, and secure authentication.

2. SmartFin Mobile — Flutter version of SmartFin
   Tech: Flutter/Dart, Provider, go_router, fl_chart, flutter_secure_storage
   What it does: Complete mobile conversion (~4,600 lines, 20 files) of SmartFin connecting to the same Next.js backend.

3. Portfolio Website (this site)
   Tech: HTML, CSS, JavaScript, Python/FastAPI (backend)
   What it does: Cinematic personal portfolio with dark/light theme, animations, AI chat.

4. PY.HUNT — Job Tracking Dashboard
   Tech: Vanilla HTML/CSS/JS, localStorage
   What it does: Browser-based dashboard to track Python developer job applications in London, with pre-built search URLs across UK job platforms.

5. Community Wellbeing Hub
   Tech: HTML, CSS, JavaScript
   What it does: Accessibility-focused community platform prototype with ARIA, focus traps, and XSS-safe rendering.

6. AI Hotel Receptionist (concept)
   Tech: Whisper STT, Claude/GPT reasoning, TTS
   What it does: Voice-based AI receptionist that communicates in Nepali — still in planning/research phase.

EDUCATION:
- Currently studying Computing / Software-related degree (2025–present)
- Completed Higher Secondary Education
- Continuous self-learning through personal projects

CONTACT:
- Open to: DMs on LinkedIn, emails, or using the contact form on this site
- Response time: Usually within 24 hours

PERSONALITY HINTS:
- Passionate about building visually immersive, cinematic digital experiences
- Interested in SaaS products and entrepreneurial ideas
- Strong interest in AI integrations in real products
- Disciplined learner who values clean code and structure
"""

# ─── Models ────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    message: str
    history: Optional[list] = []   # list of {"role": "user"/"assistant", "content": "..."}

class ContactForm(BaseModel):
    name: str
    email: str
    message: str

# ─── Health check ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Laxman Portfolio API is running"}

# ─── CHAT ENDPOINT ─────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(body: ChatMessage):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(body.message) > 500:
        raise HTTPException(status_code=400, detail="Message too long")

    # Build conversation history for Claude (multi-turn support)
    messages = []
    for h in (body.history or [])[-10:]:   # last 10 messages to stay in context
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    # Add latest user message
    messages.append({"role": "user", "content": body.message})

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=PORTFOLIO_SYSTEM_PROMPT,
            messages=messages,
        )
        reply = response.content[0].text
        return {"reply": reply, "status": "ok"}

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key — set ANTHROPIC_API_KEY")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit hit, try again shortly")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


# ─── CONTACT FORM ENDPOINT ──────────────────────────────────────────────────
@app.post("/api/contact")
async def contact(form: ContactForm):
    if not form.name.strip() or not form.message.strip():
        raise HTTPException(status_code=400, detail="Name and message are required")

    # Save to local file (always works, no email config needed)
    log_contact(form)

    # Optional: send email notification (only if EMAIL_* env vars are set)
    email_sent = False
    if os.environ.get("EMAIL_USER") and os.environ.get("EMAIL_PASS"):
        email_sent = send_email_notification(form)

    return {
        "status": "ok",
        "message": "Thanks for reaching out! Laxman will get back to you within 24 hours.",
        "email_sent": email_sent
    }


# ─── Helpers ───────────────────────────────────────────────────────────────
def log_contact(form: ContactForm):
    """Save contact submissions to a JSON log file."""
    log_path = "contacts.json"
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append({
        "timestamp": datetime.now().isoformat(),
        "name": form.name,
        "email": form.email,
        "message": form.message
    })

    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)


def send_email_notification(form: ContactForm) -> bool:
    """Send an email to yourself when someone fills the contact form."""
    try:
        smtp_user  = os.environ["EMAIL_USER"]   # e.g. your Gmail
        smtp_pass  = os.environ["EMAIL_PASS"]   # Gmail App Password
        to_address = os.environ.get("EMAIL_TO", smtp_user)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"📬 Portfolio contact from {form.name}"
        msg["From"]    = smtp_user
        msg["To"]      = to_address

        html_body = f"""
        <h2>New portfolio message</h2>
        <p><b>Name:</b> {form.name}</p>
        <p><b>Email:</b> {form.email}</p>
        <p><b>Message:</b><br>{form.message}</p>
        <hr><small>Sent via portfolio contact form</small>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_address, msg.as_string())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False
