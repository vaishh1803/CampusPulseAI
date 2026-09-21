"""CampusPulse AI API. Run: cd backend && uvicorn main:app --reload  (docs at /docs)"""
import json
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agents
import db

logging.basicConfig(level=logging.INFO)
db.init()
agents.seed_kb()
app = FastAPI(title="CampusPulse AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATUSES = ["New", "Assigned", "In Progress", "Resolved"]
SAMPLES = ["Water leaking near B block stairs", "water leak in B block staircase, floor is slippery",
           "Pipe leaking B block stairs", "Classroom fan not working in C block", "fan not working C block classroom",
           "Dustbin overflowing near canteen", "garbage overflowing canteen dustbin",
           "Projector not working in A block seminar hall", "Wifi not working in hostel", "no internet in hostel wifi",
           "Sparks from socket in D block lab", "AC not cooling in library"]


class ComplaintIn(BaseModel):
    text: str = Field(min_length=5, max_length=500)


class StatusIn(BaseModel):
    status: str


@app.post("/complaints")
def submit(c: ComplaintIn):
    return agents.run_pipeline(c.text)


@app.get("/issues")
def list_issues(status: str | None = None, priority: str | None = None):
    q, a = "SELECT i.*, (SELECT COUNT(*) FROM complaints c WHERE c.issue_id=i.id) AS reports FROM issues i WHERE 1=1", []
    if status:
        q, a = q + " AND status=?", a + [status]
    if priority:
        q, a = q + " AND priority=?", a + [priority]
    q += " ORDER BY CASE priority WHEN 'Critical' THEN 0 WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, score DESC"
    return [dict(r) for r in db.conn().execute(q, a)]


@app.get("/issues/{iid}")
def get_issue(iid: int):
    c = db.conn()
    i = c.execute("SELECT * FROM issues WHERE id=?", (iid,)).fetchone()
    if not i:
        raise HTTPException(404, "Issue not found")
    out = dict(i)
    out["factors"] = json.loads(out["factors"] or "{}")
    out["complaints"] = [dict(r) for r in c.execute("SELECT text, sim, ts FROM complaints WHERE issue_id=?", (iid,))]
    return out


@app.patch("/issues/{iid}/status")
def set_status(iid: int, s: StatusIn):
    if s.status not in STATUSES:
        raise HTTPException(422, f"status must be one of {STATUSES}")
    c = db.conn()
    c.execute("UPDATE issues SET status=? WHERE id=?", (s.status, iid))
    c.commit()
    return {"id": iid, "status": s.status}


@app.get("/analytics/summary")
def analytics():
    c = db.conn()
    n = c.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    u = c.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    crit = c.execute("SELECT COUNT(*) FROM issues WHERE priority='Critical' AND status!='Resolved'").fetchone()[0]
    return {"complaints": n, "issues": u, "duplicate_reduction_pct": round(100 * (1 - u / n), 1) if n else 0,
            "critical_open": crit}


@app.get("/insights/weekly")
def insights():
    return {"insights": agents.patterns()}


@app.post("/admin/seed")
def seed():
    return {"processed": [agents.run_pipeline(t)["issue_id"] for t in SAMPLES]}


app.mount("/app", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend"), html=True), name="app")
