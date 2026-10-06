"""
Phase 5 - FastAPI Backend
Handles: auth, document upload, session management, querying
"""

import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
service_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="NoHallucination API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class SessionCreateRequest(BaseModel):
    name: Optional[str] = "New Session"

class SessionDocRequest(BaseModel):
    document_id: str

class QueryRequest(BaseModel):
    session_id: str
    question: str


def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    try:
        response = supabase.auth.get_user(token)
        user = response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


@app.post("/auth/signup")
def signup(body: SignupRequest):
    try:
        response = supabase.auth.sign_up({
            "email": body.email,
            "password": body.password,
        })
        return {
            "user_id": response.user.id,
            "email": response.user.email,
            "access_token": response.session.access_token if response.session else None,
            "message": "Signup successful. Check your email to confirm your account."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login")
def login(body: LoginRequest):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })
        return {
            "user_id": response.user.id,
            "email": response.user.email,
            "access_token": response.session.access_token,
            "message": "Login successful"
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid email or password")


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    from ingestion.ingest import ingest_file_for_user
    user_id = user.id
    collection_name = f"user_{user_id}_docs"
    try:
        file_bytes = await file.read()
        file_name = file.filename
        file_type = file_name.split(".")[-1].lower()
        document_id = str(uuid.uuid4())
        ingest_file_for_user(
            file_bytes=file_bytes,
            file_name=file_name,
            file_type=file_type,
            collection_name=collection_name,
            document_id=document_id,
        )
        service_client.table("documents").insert({
            "id": document_id,
            "user_id": user_id,
            "filename": file_name,
            "file_type": file_type,
            "collection_name": collection_name,
        }).execute()
        return {
            "document_id": document_id,
            "filename": file_name,
            "collection": collection_name,
            "message": "Document uploaded and indexed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents(user=Depends(get_current_user)):
    user_id = user.id
    response = service_client.table("documents").select("*").eq("user_id", user_id).execute()
    return {"documents": response.data}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, user=Depends(get_current_user)):
    from retrieval.retriever import delete_document_from_qdrant
    user_id = user.id
    doc = service_client.table("documents").select("*").eq("id", document_id).eq("user_id", user_id).execute()
    if not doc.data:
        raise HTTPException(status_code=404, detail="Document not found")
    collection_name = doc.data[0]["collection_name"]
    try:
        delete_document_from_qdrant(collection_name=collection_name, document_id=document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete from vector store: {e}")
    service_client.table("documents").delete().eq("id", document_id).execute()
    return {"message": f"Document {document_id} deleted"}


@app.post("/sessions")
def create_session(body: SessionCreateRequest, user=Depends(get_current_user)):
    user_id = user.id
    response = service_client.table("sessions").insert({
        "user_id": user_id,
        "name": body.name,
    }).execute()
    return {"session": response.data[0]}


@app.get("/sessions")
def list_sessions(user=Depends(get_current_user)):
    user_id = user.id
    response = service_client.table("sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return {"sessions": response.data}


@app.get("/sessions/{session_id}")
def get_session(session_id: str, user=Depends(get_current_user)):
    user_id = user.id
    session = service_client.table("sessions").select("*").eq("id", session_id).eq("user_id", user_id).execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")
    session_docs = service_client.table("session_documents").select(
        "document_id, documents(filename, file_type, uploaded_at)"
    ).eq("session_id", session_id).execute()
    queries = service_client.table("queries").select("*").eq("session_id", session_id).order("created_at").execute()
    return {
        "session": session.data[0],
        "active_documents": session_docs.data,
        "query_history": queries.data,
    }


@app.post("/sessions/{session_id}/documents")
def add_document_to_session(session_id: str, body: SessionDocRequest, user=Depends(get_current_user)):
    user_id = user.id
    session = service_client.table("sessions").select("id").eq("id", session_id).eq("user_id", user_id).execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")
    doc = service_client.table("documents").select("id").eq("id", body.document_id).eq("user_id", user_id).execute()
    if not doc.data:
        raise HTTPException(status_code=404, detail="Document not found")
    service_client.table("session_documents").upsert({
        "session_id": session_id,
        "document_id": body.document_id,
    }).execute()
    return {"message": "Document added to session"}


@app.delete("/sessions/{session_id}/documents/{document_id}")
def remove_document_from_session(session_id: str, document_id: str, user=Depends(get_current_user)):
    user_id = user.id
    session = service_client.table("sessions").select("id").eq("id", session_id).eq("user_id", user_id).execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")
    service_client.table("session_documents").delete().eq("session_id", session_id).eq("document_id", document_id).execute()
    return {"message": "Document removed from session"}


@app.post("/query")
def run_query(body: QueryRequest, user=Depends(get_current_user)):
    from notebooks.pipeline import run_pipeline
    user_id = user.id
    session = service_client.table("sessions").select("id").eq("id", body.session_id).eq("user_id", user_id).execute()
    if not session.data:
        raise HTTPException(status_code=404, detail="Session not found")
    session_docs = service_client.table("session_documents").select("document_id").eq("session_id", body.session_id).execute()
    active_doc_ids = [row["document_id"] for row in session_docs.data]
    if not active_doc_ids:
        raise HTTPException(status_code=400, detail="No documents added to this session.")
    collection_name = f"user_{user_id}_docs"
    try:
        result = run_pipeline(
            query=body.question,
            collection_name=collection_name,
            document_ids=active_doc_ids,
        )
        query_record = {
            "session_id": body.session_id,
            "question": body.question,
            "answer": result.get("final_answer") or result.get("draft_answer"),
            "pii_detected": result.get("pii_detected", False),
            "is_toxic": result.get("is_toxic", False),
            "judge_scores": result.get("judge_scores"),
            "judge_overall": result.get("judge_overall"),
            "blocked": result.get("blocked", False),
            "block_reason": result.get("block_reason"),
        }
        saved = service_client.table("queries").insert(query_record).execute()
        return {
            "query_id": saved.data[0]["id"],
            "answer": query_record["answer"],
            "blocked": query_record["blocked"],
            "block_reason": query_record["block_reason"],
            "judge_scores": query_record["judge_scores"],
            "judge_overall": query_record["judge_overall"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health():
    return {"status": "ok", "service": "NoHallucination API"}