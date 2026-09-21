import os
import sys
import tempfile

d = tempfile.mkdtemp()
os.environ.update(MOCK_LLM="1", DB_PATH=f"{d}/t.db", CHROMA_PATH=f"{d}/chroma")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import agents  # noqa: E402
import db  # noqa: E402

db.init()


def test_duplicates_merge_into_one_issue():
    a = agents.run_pipeline("Water leaking near B block stairs")
    b = agents.run_pipeline("water leak in B block staircase, floor slippery")
    assert a["issue_id"] == b["issue_id"]


def test_different_block_is_not_merged():
    a = agents.run_pipeline("Fan not working in C block classroom")
    b = agents.run_pipeline("Fan not working in A block classroom")
    assert a["issue_id"] != b["issue_id"]


def test_safety_override_is_critical():
    r = agents.run_pipeline("Sparks from socket in D block lab")
    row = db.conn().execute("SELECT priority FROM issues WHERE id=?", (r["issue_id"],)).fetchone()
    assert row["priority"] == "Critical"
