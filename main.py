"""
French Trainer Backend - FastAPI Server
Handles question generation, AI mode, and user progress tracking.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "french_trainer.db")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ── Database Setup ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'default',
            topic_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            correct INTEGER NOT NULL,
            question_type TEXT,
            answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_stats (
            user_id TEXT PRIMARY KEY DEFAULT 'default',
            xp INTEGER DEFAULT 0,
            streak INTEGER DEFAULT 0,
            last_practice_date TEXT,
            total_answered INTEGER DEFAULT 0,
            total_correct INTEGER DEFAULT 0,
            vocab_learned INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_progress_user ON user_progress(user_id);
        CREATE INDEX IF NOT EXISTS idx_progress_topic ON user_progress(topic_id);
    """)
    conn.commit()
    conn.close()


# ── Lifespan ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# ── App Setup ───────────────────────────────────────────────────
app = FastAPI(
    title="French Trainer API",
    description="Backend API for the French Trainer TCF Practice Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────
class AnswerSubmission(BaseModel):
    topic_id: str
    question_id: str
    correct: bool
    question_type: str


class AIQuestionRequest(BaseModel):
    topic: str
    level: str  # A1, A2, B1, B2
    question_type: str  # mcq, fill, vocab, pronunciation
    count: int = 3


class AIQuestion(BaseModel):
    id: str
    type: str
    level: str
    question: Optional[str] = None
    options: Optional[list] = None
    correct: Optional[int] = None
    answer: Optional[str] = None
    hint: Optional[str] = None
    explanation: Optional[str] = None
    word: Optional[str] = None
    meaning: Optional[str] = None
    context: Optional[str] = None
    contextTranslation: Optional[str] = None
    text: Optional[str] = None
    translation: Optional[str] = None
    phonetic: Optional[str] = None


# ── Routes ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "French Trainer API", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ── Progress Endpoints ──────────────────────────────────────────
@app.post("/api/progress/answer")
async def submit_answer(submission: AnswerSubmission):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_progress (topic_id, question_id, correct, question_type) VALUES (?, ?, ?, ?)",
        (submission.topic_id, submission.question_id, int(submission.correct), submission.question_type),
    )

    # Update user stats
    cursor.execute("SELECT * FROM user_stats WHERE user_id = 'default'")
    stats = cursor.fetchone()
    today = datetime.now().strftime("%Y-%m-%d")
    xp_gain = 10 if submission.correct else 2

    if stats:
        last_date = stats["last_practice_date"]
        streak = stats["streak"]
        if last_date != today:
            # Check if yesterday
            from datetime import timedelta
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            streak = streak + 1 if last_date == yesterday else 1

        cursor.execute(
            """UPDATE user_stats SET xp = xp + ?, streak = ?, last_practice_date = ?,
               total_answered = total_answered + 1, total_correct = total_correct + ?,
               vocab_learned = vocab_learned + ?, updated_at = CURRENT_TIMESTAMP
               WHERE user_id = 'default'""",
            (xp_gain, streak, today, int(submission.correct),
             1 if submission.question_type == "vocab" and submission.correct else 0),
        )
    else:
        cursor.execute(
            """INSERT INTO user_stats (user_id, xp, streak, last_practice_date, total_answered, total_correct, vocab_learned)
               VALUES ('default', ?, 1, ?, 1, ?, ?)""",
            (xp_gain, today, int(submission.correct),
             1 if submission.question_type == "vocab" and submission.correct else 0),
        )

    conn.commit()
    conn.close()
    return {"success": True, "xp_gained": xp_gain}


@app.get("/api/progress/stats")
async def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_stats WHERE user_id = 'default'")
    stats = cursor.fetchone()
    conn.close()

    if not stats:
        return {"xp": 0, "streak": 0, "total_answered": 0, "total_correct": 0, "vocab_learned": 0}

    return dict(stats)


@app.get("/api/progress/topic/{topic_id}")
async def get_topic_progress(topic_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) as total, SUM(correct) as correct FROM user_progress WHERE topic_id = ? AND user_id = 'default'",
        (topic_id,),
    )
    result = cursor.fetchone()
    conn.close()

    return {
        "topic_id": topic_id,
        "total_answered": result["total"],
        "total_correct": result["correct"] or 0,
        "accuracy": round((result["correct"] or 0) / max(result["total"], 1) * 100),
    }


# ── AI Question Generation ─────────────────────────────────────
@app.post("/api/ai/generate")
async def generate_ai_questions(request: AIQuestionRequest):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise HTTPException(status_code=400, detail="OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env")

    prompt = _build_prompt(request.topic, request.level, request.question_type, request.count)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": [
                        {"role": "system", "content": "You are a French language teacher creating practice questions for TCF exam preparation. Always respond with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.8,
                    "max_tokens": 2000,
                },
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        # Try to parse JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            questions = json.loads(content.strip())
            if isinstance(questions, dict) and "questions" in questions:
                questions = questions["questions"]
            return {"questions": questions}
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"OpenRouter API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


def _build_prompt(topic: str, level: str, q_type: str, count: int) -> str:
    type_instructions = {
        "mcq": """Generate multiple choice questions with this JSON format:
[{"id": "ai_1", "type": "mcq", "level": "%LEVEL%", "question": "question text in French", "options": ["opt1", "opt2", "opt3", "opt4"], "correct": 0, "explanation": "explanation in English"}]""",
        "fill": """Generate fill-in-the-blank questions with this JSON format:
[{"id": "ai_1", "type": "fill", "level": "%LEVEL%", "question": "sentence with _____ blank", "answer": "correct word", "hint": "helpful hint", "explanation": "explanation in English"}]""",
        "vocab": """Generate vocabulary flashcard questions with this JSON format:
[{"id": "ai_1", "type": "vocab", "level": "%LEVEL%", "word": "French word", "meaning": "English meaning", "context": "example sentence in French", "contextTranslation": "English translation"}]""",
        "pronunciation": """Generate pronunciation practice items with this JSON format:
[{"id": "ai_1", "type": "pronunciation", "level": "%LEVEL%", "text": "French sentence", "translation": "English translation", "phonetic": "IPA phonetic transcription"}]""",
    }

    instruction = type_instructions.get(q_type, type_instructions["mcq"]).replace("%LEVEL%", level)

    return f"""Create {count} French language practice questions about "{topic}" at CEFR level {level} for TCF exam preparation.

{instruction}

Requirements:
- Questions must be appropriate for {level} level
- Use natural, real-world French
- Provide clear explanations
- Make questions progressively challenging
- Only output valid JSON array, no other text"""


# ── Run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
