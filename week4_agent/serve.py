# week4_agent/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="Agent API")

class Query(BaseModel):
    question: str
    max_steps: int = 6

class Answer(BaseModel):
    answer: str
    steps: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "empty question")
    if q.max_steps > 12:
        raise HTTPException(400, "max_steps too large")
    answer, trace = run_agent(q.question, max_steps=q.max_steps, verbose=False)
    return Answer(answer=answer, steps=sum(1 for m in trace if m["role"] == "tool"))
