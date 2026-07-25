import uuid
from sqlalchemy import Column, String, Text, ForeignKey, Integer, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Skill(Base):
    __tablename__ = "skills"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    steps = relationship("RoadmapStep", back_populates="skill", cascade="all, delete-orphan")
    saved_by_users = relationship("UserSavedSkill", back_populates="skill", cascade="all, delete-orphan")

class RoadmapStep(Base):
    __tablename__ = "roadmap_steps"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=False)
    order = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)

    skill = relationship("Skill", back_populates="steps")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)

    session = relationship("ChatSession", back_populates="messages")

class SuggestedSkillLog(Base):
    """Tracks every skill suggested by Groq AI across chat sessions"""
    __tablename__ = "suggested_skill_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    skill = relationship("Skill")

class UserSavedSkill(Base):
    """Tracks bookmarked/favorite skills for a user"""
    __tablename__ = "user_saved_skills"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(50), nullable=False, default="default_user")  # Can tie to auth later
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=False)
    saved_at = Column(DateTime, default=datetime.utcnow)

    skill = relationship("Skill", back_populates="saved_by_users")
