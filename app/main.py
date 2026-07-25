import os
import re
import uuid
import json
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from groq import Groq

# Direct imports from the same directory
from database import engine, Base, get_db
from models import Skill, RoadmapStep, SuggestedSkillLog, UserSavedSkill
from schemas import (
    ChatRequest, 
    ChatResponse, 
    SuggestedSkill, 
    SkillRoadmapResponse, 
    RoadmapStepSchema,
    SaveSkillRequest,
    SaveSkillResponse,
    SavedSkillItem
)
from services import process_chat_session

# -------------------------------------------------------------------
# Environment & LLM Client Setup
# -------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Skill Guidance Chatbot API",
    description="APIs for skill guidance, chat interaction, YouTube video roadmap analysis, and coding challenge generation.",
    version="2.1.0"
)

# -------------------------------------------------------------------
# Database Startup Seeding
# -------------------------------------------------------------------
@app.on_event("startup")
def seed_database():
    db = next(get_db())
    if db.query(Skill).count() == 0:
        rn_id = str(uuid.uuid4())
        fa_id = str(uuid.uuid4())

        rn_skill = Skill(id=rn_id, name="React Native Mobile Development", description="Build cross-platform iOS/Android apps.")
        fa_skill = Skill(id=fa_id, name="FastAPI Backend Engineering", description="Build high-performance RESTful Python APIs.")

        db.add_all([rn_skill, fa_skill])
        db.commit()

        db.add_all([
            RoadmapStep(skill_id=rn_id, order=1, title="JavaScript & TypeScript Fundamentals", description="Master ES6+, async/await, and TypeScript types."),
            RoadmapStep(skill_id=rn_id, order=2, title="React Core & Hooks", description="Learn state management, useEffect, custom hooks, and component lifecycle."),
            RoadmapStep(skill_id=rn_id, order=3, title="React Native Components & Layout", description="Build layouts using Flexbox, View, Text, FlatList, and StyleSheet."),
            RoadmapStep(skill_id=rn_id, order=4, title="Offline-First Architecture", description="Implement local SQLite storage, sync logic, and state preservation."),
        ])

        db.add_all([
            RoadmapStep(skill_id=fa_id, order=1, title="Python 3.11+ & Pydantic", description="Master type hinting, dataclasses, and Pydantic schema validation."),
            RoadmapStep(skill_id=fa_id, order=2, title="Async FastAPI Core", description="Understand routing, dependency injection, and middleware."),
            RoadmapStep(skill_id=fa_id, order=3, title="Database Integration with SQLAlchemy", description="Setup SQLite ORM models and migrations."),
        ])
        db.commit()

# -------------------------------------------------------------------
# Helper Functions for YouTube & Groq
# -------------------------------------------------------------------
def extract_youtube_id(url_or_id: str) -> str:
    """Extract 11-character YouTube video ID from various URL formats."""
    regex = r"(?:v=|\/([0-9A-Za-z_-]{11}).*|list=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url_or_id)
    if match:
        return match.group(2) if match.group(2) else match.group(1)
    if len(url_or_id) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id
    raise HTTPException(status_code=400, detail="Invalid YouTube Video URL or ID.")


def fetch_youtube_transcript(video_id: str) -> str:
    """Retrieves and concatenates transcript text for a YouTube video."""
    try:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        except AttributeError:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'en-US'])
            
        full_text = " ".join([item['text'] for item in transcript_list])
        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(status_code=404, detail="No public English transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")


def get_active_groq_client() -> Groq:
    """Ensure Groq client is properly initialized."""
    if not groq_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is missing in your environment configuration (.env)."
        )
    return groq_client

# -------------------------------------------------------------------
# Request / Response Schemas for Roadmap Tools
# -------------------------------------------------------------------
class AnalyzeVideoRequest(BaseModel):
    roadmap_id: str = Field(..., example="roadmap_react_native_101")
    step_title: str = Field(..., example="State Management with Redux Toolkit")
    step_description: Optional[str] = Field(None, example="Understanding slices, store setup, and useDispatch/useSelector hooks.")
    youtube_url: str = Field(..., example="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

class LessonItem(BaseModel):
    title: str
    takeaway: str

class AnalyzeVideoResponse(BaseModel):
    roadmap_id: str
    step_title: str
    youtube_id: str
    is_relevant: bool
    relevance_score: int = Field(..., description="Relevance score from 0 to 100")
    key_lessons: List[LessonItem]
    summary: str

class ChallengeRequest(BaseModel):
    roadmap_id: str = Field(..., example="roadmap_react_native_101")
    step_title: str = Field(..., example="State Management with Redux Toolkit")
    difficulty: Optional[str] = Field("Medium", example="Medium")
    video_summary_or_transcript: Optional[str] = Field(None, description="Optional text context if video transcript was already processed.")
    youtube_url: Optional[str] = Field(None, description="Provide if transcript needs to be fetched on the fly.")

class CodingChallengeResponse(BaseModel):
    roadmap_id: str
    step_title: str
    challenge_title: str
    difficulty: str
    problem_statement: str
    starter_code: str
    hints: List[str]
    expected_output_or_criteria: str

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

# --- Endpoint 1: Chat API ---
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, db: Session = Depends(get_db)):
    session_id, llm_output = process_chat_session(payload.session_id, payload.message, db)

    suggested_skills: List[SuggestedSkill] = []

    if llm_output.is_finished and llm_output.recommended_skill_ids:
        skills = db.query(Skill).filter(Skill.id.in_(llm_output.recommended_skill_ids)).all()
        suggested_skills = [SuggestedSkill.model_validate(s) for s in skills]

    return ChatResponse(
        session_id=session_id,
        response=llm_output.reply,
        is_finished=llm_output.is_finished,
        suggested_skills=suggested_skills
    )

# --- Endpoint 2: Skill Roadmap API ---
@app.get("/api/skills/{skill_id}/roadmap", response_model=SkillRoadmapResponse)
def get_skill_roadmap(skill_id: str, db: Session = Depends(get_db)):
    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Skill with UUID '{skill_id}' not found."
        )

    steps = (
        db.query(RoadmapStep)
        .filter(RoadmapStep.skill_id == skill_id)
        .order_by(RoadmapStep.order.asc())
        .all()
    )

    return SkillRoadmapResponse(
        skill_id=skill.id,
        skill_name=skill.name,
        roadmap=[RoadmapStepSchema.model_validate(step) for step in steps]
    )

# --- Endpoint 3: Get All AI Suggested Skills ---
@app.get("/api/skills/suggested", response_model=List[SuggestedSkill])
def get_all_suggested_skills(session_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Skill).join(SuggestedSkillLog, Skill.id == SuggestedSkillLog.skill_id)
    
    if session_id:
        query = query.filter(SuggestedSkillLog.session_id == session_id)

    suggested_skills = query.distinct().all()
    return [SuggestedSkill.model_validate(s) for s in suggested_skills]

# --- Endpoint 4: Save Favorite Skill ---
@app.post("/api/skills/save", response_model=SaveSkillResponse)
def save_favorite_skill(payload: SaveSkillRequest, db: Session = Depends(get_db)):
    skill = db.query(Skill).filter(Skill.id == payload.skill_id).first()
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill with UUID '{payload.skill_id}' does not exist."
        )

    existing = (
        db.query(UserSavedSkill)
        .filter(
            UserSavedSkill.skill_id == payload.skill_id,
            UserSavedSkill.user_id == payload.user_id
        )
        .first()
    )

    if existing:
        return SaveSkillResponse(
            message="Skill is already in favorites.",
            saved_skill_id=existing.id,
            skill=SuggestedSkill.model_validate(skill)
        )

    new_saved = UserSavedSkill(skill_id=payload.skill_id, user_id=payload.user_id)
    db.add(new_saved)
    db.commit()
    db.refresh(new_saved)

    return SaveSkillResponse(
        message="Skill successfully saved to favorites.",
        saved_skill_id=new_saved.id,
        skill=SuggestedSkill.model_validate(skill)
    )

# --- Endpoint 5: Get Saved/Favorite Skills ---
@app.get("/api/skills/saved", response_model=List[SavedSkillItem])
def get_saved_skills(user_id: Optional[str] = "default_user", db: Session = Depends(get_db)):
    """
    Retrieves all skills favorited/saved by a specific user.
    """
    saved_entries = (
        db.query(UserSavedSkill)
        .filter(UserSavedSkill.user_id == user_id)
        .order_by(UserSavedSkill.saved_at.desc())
        .all()
    )

    return [
        SavedSkillItem(
            saved_id=item.id,
            user_id=item.user_id,
            saved_at=item.saved_at,
            skill=SuggestedSkill.model_validate(item.skill)
        )
        for item in saved_entries
    ]

# --- Endpoint 6: Analyze Video for Roadmap Step ---
@app.post("/api/v1/roadmap/analyze-video", response_model=AnalyzeVideoResponse)
async def analyze_video_for_step(payload: AnalyzeVideoRequest):
    """
    Checks if a YouTube video aligns with a specific roadmap step,
    extracts core lessons, and structures the findings.
    """
    client = get_active_groq_client()
    video_id = extract_youtube_id(payload.youtube_url)
    transcript = fetch_youtube_transcript(video_id)
    
    truncated_transcript = transcript[:12000]

    system_prompt = (
        "You are an expert technical curriculum builder. "
        "Analyze the provided transcript against a specific roadmap learning step. "
        "Return ONLY a raw JSON object with the following keys: "
        "'is_relevant' (boolean), 'relevance_score' (int 0-100), "
        "'summary' (string), and 'key_lessons' (list of objects with 'title' and 'takeaway')."
    )

    user_prompt = f"""
    Roadmap Step: {payload.step_title}
    Step Description: {payload.step_description or 'N/A'}
    
    Video Transcript Excerpt:
    "{truncated_transcript}"
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)

        return AnalyzeVideoResponse(
            roadmap_id=payload.roadmap_id,
            step_title=payload.step_title,
            youtube_id=video_id,
            is_relevant=result.get("is_relevant", False),
            relevance_score=result.get("relevance_score", 0),
            key_lessons=[LessonItem(**item) for item in result.get("key_lessons", [])],
            summary=result.get("summary", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

# --- Endpoint 7: Generate Coding Challenge ---
@app.post("/api/v1/roadmap/generate-challenge", response_model=CodingChallengeResponse)
async def generate_coding_challenge(payload: ChallengeRequest):
    """
    Generates a coding challenge tailored to the topic and content of the video.
    """
    client = get_active_groq_client()
    context_text = ""

    if payload.video_summary_or_transcript:
        context_text = payload.video_summary_or_transcript[:12000]
    elif payload.youtube_url:
        video_id = extract_youtube_id(payload.youtube_url)
        context_text = fetch_youtube_transcript(video_id)[:12000]
    else:
        context_text = f"General concepts around: {payload.step_title}"

    system_prompt = (
        "You are an expert technical interviewer and coding instructor. "
        "Create a practical coding challenge based on the topic and video context provided. "
        "Return ONLY a raw JSON object with the following keys: "
        "'challenge_title' (string), 'problem_statement' (string), "
        "'starter_code' (string), 'hints' (list of strings), "
        "and 'expected_output_or_criteria' (string)."
    )

    user_prompt = f"""
    Roadmap Step: {payload.step_title}
    Difficulty: {payload.difficulty}
    Context/Content:
    "{context_text}"
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4
        )

        result = json.loads(response.choices[0].message.content)

        return CodingChallengeResponse(
            roadmap_id=payload.roadmap_id,
            step_title=payload.step_title,
            difficulty=payload.difficulty or "Medium",
            challenge_title=result.get("challenge_title", "Hands-on Exercise"),
            problem_statement=result.get("problem_statement", ""),
            starter_code=result.get("starter_code", ""),
            hints=result.get("hints", []),
            expected_output_or_criteria=result.get("expected_output_or_criteria", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")
