"""CampusPulse agents: Intake -> Dedup -> RAG -> Priority -> Summary, plus a Pattern agent."""
import json
import logging
import os
import re
import time

import chromadb

from db import conn
from llm import chat_json, embed

log = logging.getLogger("campuspulse")
HIGH = float(os.getenv("SIM_HIGH", "0.85"))  # auto-merge above this similarity
LOW = float(os.getenv("SIM_LOW", "0.60"))    # between LOW and HIGH, Granite confirms
CATS = ["Electrical", "Plumbing", "Sanitation", "IT/AV", "Furniture", "Safety", "Other"]
TEAM = {"Electrical": "Electrical Maintenance", "Plumbing": "Plumbing and Civil", "Sanitation": "Housekeeping",
        "IT/AV": "IT Support", "Furniture": "Carpentry", "Safety": "Safety Officer", "Other": "Admin Office"}
KW = {"Electrical": "fan light ac socket switch wire power", "Plumbing": "water leak tap pipe flush drain",
      "Sanitation": "dustbin garbage trash smell toilet", "IT/AV": "projector wifi internet computer screen",
      "Furniture": "bench chair desk table board door"}
LOC = {"lab": 1.0, "hostel": 1.0, "library": 1.0, "seminar hall": 1.0, "classroom": 0.6}
W = {"safety": .3, "reports": .25, "time": .15, "recurrence": .15, "location": .15}
STRONG = r"spark|shock|fire|smoke|exposed|short circuit"

# Sample knowledge base. Replace with your college's real SOPs and contacts.
KB = ["Electrical faults: isolate power and call Electrical Maintenance. Sparks or shocks are emergencies.",
      "Plumbing leaks: close the nearest valve, place a wet-floor sign, inform Plumbing and Civil.",
      "Sanitation: overflowing bins must be cleared within 4 hours by Housekeeping.",
      "IT/AV: projector, Wi-Fi and lab computer faults go to IT Support. Hostel Wi-Fi within 24 hours.",
      "Safety: fire, smoke or structural cracks go to the Safety Officer immediately.",
      "Escalation: an issue open for more than 72 hours or with 10+ reports goes to the Estate Officer."]

_chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "chroma_db"))
issues_col = _chroma.get_or_create_collection("issues", metadata={"hnsw:space": "cosine"})
kb_col = _chroma.get_or_create_collection("campus_kb", metadata={"hnsw:space": "cosine"})


def seed_kb():
    if kb_col.count() == 0:
        for i, doc in enumerate(KB):
            v = embed(doc)
            if v:
                kb_col.add(ids=[f"kb{i}"], embeddings=[v], documents=[doc])


# ---- 1. Intake agent -------------------------------------------------------
def rule_intake(t):
    low = t.lower()
    cat = "Other"
    if re.search(STRONG, low):
        cat = "Safety"
    else:
        best = 0
        for k, words in KW.items():
            n = sum(bool(re.search(rf"\b{w}", low)) for w in words.split())
            if n > best:
                best, cat = n, k
    b = re.search(r"\b([a-f])\s*-?\s*block\b|\bblock\s*([a-f])\b", t, re.I)
    p = re.search(r"\b(lab|hostel|library|canteen|class(?:room)?|stair\w*|corridor|washroom|seminar hall)\b", low)
    place = p.group(1) if p else None
    if place:
        place = re.sub(r"^stair\w*", "stairs", place)
        place = "classroom" if place == "class" else place
    return {"category": cat, "block": (b.group(1) or b.group(2)).upper() if b else None,
            "place": place, "description": t.strip()}


def intake(text):
    """Granite extracts structured fields; rules fill gaps if the output is missing or invalid."""
    base = rule_intake(text)
    out = chat_json(f'Extract JSON with keys category (one of {CATS}), block (single letter or null), place '
                    f'(lab, hostel, library, canteen, classroom, stairs, corridor, washroom, seminar hall or null) '
                    f'and description (one short normalized sentence) from this campus complaint. '
                    f'It may mix Tamil and English.\nComplaint: "{text}"')
    if out and out.get("category") in CATS:
        base.update({k: out[k] for k in ("category", "block", "place", "description") if out.get(k)})
    base["block"] = str(base["block"]).upper()[:1] if base["block"] else None
    return base


# ---- 2. Deduplication agent -----------------------------------------------
def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / max(len(a | b), 1)


def same_issue(a, b):
    out = chat_json(f'Are these two campus complaints about the same underlying problem? '
                    f'Reply JSON {{"same": true or false}}.\nA: {a}\nB: {b}')
    return bool(out and out.get("same") is True)


def find_match(c):
    rows = [r for r in conn().execute("SELECT * FROM issues WHERE category=?", (c["category"],))
            if not (c["block"] and r["block"] and c["block"] != r["block"])]
    if not rows:
        return None, 0.0
    vec = embed(c["description"])
    best, score = None, 0.0
    if vec and issues_col.count():
        res = issues_col.query(query_embeddings=[vec], n_results=min(5, issues_col.count()),
                               where={"category": c["category"]})
        ok = {str(r["id"]): r for r in rows}
        for i, d in zip(res["ids"][0], res["distances"][0]):
            if i in ok and 1 - d > score:
                best, score = ok[i], 1 - d
        if best is None:
            return None, score
        if score >= HIGH or (score >= LOW and same_issue(c["description"], best["title"])):
            return best, score
        return None, score
    for r in rows:  # keyword fallback when embeddings are unavailable
        s = jaccard(c["description"], r["title"]) + (0.3 if c["block"] and c["block"] == r["block"] else 0)
        if s > score:
            best, score = r, min(s, 0.99)
    return (best, score) if score >= 0.4 else (None, score)


def attach_or_create(text, c):
    db, now = conn(), time.time()
    m, sim = find_match(c)
    if m:
        iid, created = m["id"], False
        if m["status"] == "Resolved":  # reopen when a matching complaint arrives
            db.execute("UPDATE issues SET status='New' WHERE id=?", (iid,))
    else:
        cur = db.execute("INSERT INTO issues(title,category,block,place,first_ts) VALUES(?,?,?,?,?)",
                         (c["description"], c["category"], c["block"], c["place"], now))
        iid, created, sim = cur.lastrowid, True, 1.0
        v = embed(c["description"])
        if v:
            issues_col.add(ids=[str(iid)], embeddings=[v], documents=[c["description"]],
                           metadatas=[{"category": c["category"]}])
    db.execute("INSERT INTO complaints(issue_id,text,sim,ts) VALUES(?,?,?,?)", (iid, text, sim, now))
    db.commit()
    return iid, created, sim


# ---- 3. RAG agent ----------------------------------------------------------
def rag(iid):
    db = conn()
    i = db.execute("SELECT * FROM issues WHERE id=?", (iid,)).fetchone()
    past = db.execute("SELECT COUNT(*) FROM issues WHERE category=? AND block IS ? AND id!=?",
                      (i["category"], i["block"], iid)).fetchone()[0]
    hint, v = None, embed(f'{i["category"]} problem {i["place"] or ""} procedure')
    if v and kb_col.count():
        hint = kb_col.query(query_embeddings=[v], n_results=1)["documents"][0][0]
    team = TEAM.get(i["category"], "Admin Office")
    db.execute("UPDATE issues SET recurring=?, team=? WHERE id=?", (int(past > 0), team, iid))
    db.commit()
    return {"recurring": past > 0, "similar_past_issues": past, "team": team, "sop_hint": hint}


# ---- 4. Priority agent (transparent score + LLM justification) -------------
def priority(iid):
    db = conn()
    i = db.execute("SELECT * FROM issues WHERE id=?", (iid,)).fetchone()
    texts = [r["text"] for r in db.execute("SELECT text FROM complaints WHERE issue_id=?", (iid,))]
    txt, n = " ".join(texts).lower(), len(texts)
    strong = bool(re.search(STRONG, txt))
    f = {"safety": 1.0 if strong else (0.4 if re.search(r"leak|flood|slip|crack", txt) else 0.0),
         "reports": min(n / 10, 1), "time": min((time.time() - i["first_ts"]) / 3600 / 48, 1),
         "recurrence": float(i["recurring"]), "location": LOC.get(i["place"], 0.3)}
    score = round(100 * sum(f[k] * W[k] for k in W))
    level = "Critical" if strong or score >= 60 else "High" if score >= 42 else "Medium" if score >= 25 else "Low"
    out = chat_json(f'Return JSON {{"why": "<one sentence>"}} explaining why this campus issue is {level} '
                    f'priority. Issue: {i["title"]}. Reports: {n}. Factors (0-1): {json.dumps(f)}')
    why = (out or {}).get("why") or f"{level}: {n} report(s), safety factor {f['safety']}."
    db.execute("UPDATE issues SET priority=?, score=?, factors=? WHERE id=?",
               (level, score, json.dumps({**f, "why": why}), iid))
    db.commit()
    return {"priority": level, "score": score, "factors": f, "why": why}


# ---- 5. Summary agent ------------------------------------------------------
def summarize(iid):
    db = conn()
    i = db.execute("SELECT * FROM issues WHERE id=?", (iid,)).fetchone()
    n = db.execute("SELECT COUNT(*) FROM complaints WHERE issue_id=?", (iid,)).fetchone()[0]
    out = chat_json(f'Return JSON {{"summary": "<two sentences for campus administration>"}} for: {i["title"]}, '
                    f'block {i["block"]}, {n} reports, priority {i["priority"]}, team {i["team"]}.')
    s = (out or {}).get("summary") or f'{n} report(s): {i["title"]}. Suggested action: assign to {i["team"]}.'
    db.execute("UPDATE issues SET summary=? WHERE id=?", (s, iid))
    db.commit()
    return s


def run_pipeline(text):
    """Run all agents in order and return a per-agent trace for explainability."""
    c = intake(text)
    iid, created, sim = attach_or_create(text, c)
    r, p = rag(iid), priority(iid)
    s = summarize(iid)
    trace = [{"agent": "intake", "output": c},
             {"agent": "dedup", "output": {"issue_id": iid, "created": created, "similarity": round(sim, 2)}},
             {"agent": "rag", "output": r}, {"agent": "priority", "output": p},
             {"agent": "summary", "output": s}]
    for t in trace:
        log.info("%s -> %s", t["agent"], t["output"])
    return {"issue_id": iid, "trace": trace}


# ---- Pattern agent ---------------------------------------------------------
def patterns():
    rows = conn().execute("""SELECT i.category, i.block, COUNT(c.id) AS n FROM complaints c
                             JOIN issues i ON i.id=c.issue_id GROUP BY 1,2 ORDER BY n DESC LIMIT 5""").fetchall()
    data = [dict(r) for r in rows]
    out = chat_json(f'You are a campus facilities analyst. Given complaint counts {json.dumps(data)}, '
                    f'return JSON {{"insights": ["<3 short insight strings>"]}}.')
    if out and isinstance(out.get("insights"), list):
        return out["insights"]
    return [f'{d["category"]} problems in Block {d["block"] or "unknown"}: {d["n"]} reports' for d in data[:3]]
