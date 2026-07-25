from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- Chat Schemas ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = ""

class SuggestedSkill(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    session_id: str
    response: str
    is_finished: bool
    suggested_skills: Optional[List[SuggestedSkill]] = []

# --- Groq Structured Output ---
class GroqChatLLMOutput(BaseModel):
    reply: str
    is_finished: bool
    recommended_skill_ids: Optional[List[str]] = []

# --- Roadmap Schemas ---
class RoadmapStepSchema(BaseModel):
    id: str
    order: int
    title: str
    description: str

    class Config:
        from_attributes = True

class SkillRoadmapResponse(BaseModel):
    skill_id: str
    skill_name: str
    roadmap: List[RoadmapStepSchema]

# --- Save Skill Schemas ---
class SaveSkillRequest(BaseModel):
    skill_id: str
    user_id: Optional[str] = "default_user"

class SaveSkillResponse(BaseModel):
    message: str
    saved_skill_id: str
    skill: SuggestedSkill

class SavedSkillItem(BaseModel):
    saved_id: str
    user_id: str
    saved_at: datetime
    skill: SuggestedSkill

    class Config:
        from_attributes = True