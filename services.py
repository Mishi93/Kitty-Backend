import os
import json
import uuid
from groq import Groq
from sqlalchemy.orm import Session
from app.models import Skill, ChatSession, ChatMessage, SuggestedSkillLog
from app.schemas import GroqChatLLMOutput

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def process_chat_session(session_id: str | None, user_message: str | None, db: Session):
    # 1. Retrieve or create session
    if not session_id:
        session_id = str(uuid.uuid4())
        session = ChatSession(id=session_id)
        db.add(session)
        db.commit()
    else:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(id=session_id)
            db.add(session)
            db.commit()

    # 2. Fetch conversation history
    past_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )

    # 3. Fetch available skills from DB to provide to LLM context
    available_skills = db.query(Skill).all()
    skills_context = ", ".join([f"UUID '{s.id}': {s.name}" for s in available_skills])

    system_prompt = f"""
    You are an intelligent career and skill counseling chatbot.
    Your task is to converse naturally with the user, understand their goals, and assess whether you have gathered enough context to recommend skill learning paths.

    Available Skills in database:
    [{skills_context}]

    Rules:
    1. If user message is empty and history is empty, welcome them warmly and ask about their goals.
    2. If you need more context, set `is_finished` to false, keep `recommended_skill_ids` empty, and give a `reply`.
    3. When you have sufficient information or user asks for recommendations, set `is_finished` to true, provide a concluding `reply`, and select matching skill UUID strings in `recommended_skill_ids`.

    You MUST return ONLY a valid JSON object matching this schema:
    {{
      "reply": "string message to user",
      "is_finished": true/false,
      "recommended_skill_ids": ["uuid-string-1", "uuid-string-2"]
    }}
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in past_messages:
        messages.append({"role": msg.role, "content": msg.content})

    actual_message = user_message.strip() if user_message else "Hello! Starting new session."
    messages.append({"role": "user", "content": actual_message})

    if user_message and user_message.strip():
        db.add(ChatMessage(session_id=session_id, role="user", content=user_message.strip()))
        db.commit()

    # 4. Call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3
    )

    raw_content = response.choices[0].message.content
    data = json.loads(raw_content)

    llm_output = GroqChatLLMOutput(
        reply=data.get("reply", ""),
        is_finished=data.get("is_finished", False),
        recommended_skill_ids=data.get("recommended_skill_ids", [])
    )

    # Save assistant response
    db.add(ChatMessage(session_id=session_id, role="assistant", content=llm_output.reply))

    # Log suggested skills to DB if finished
    if llm_output.is_finished and llm_output.recommended_skill_ids:
        for sk_id in llm_output.recommended_skill_ids:
            db.add(SuggestedSkillLog(session_id=session_id, skill_id=sk_id))

    db.commit()

    return session_id, llm_output