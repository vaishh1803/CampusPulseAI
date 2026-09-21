"""SQLite storage: one row per underlying issue, one row per raw complaint."""
import os
import sqlite3

DB = os.getenv("DB_PATH", "campuspulse.db")


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS issues(
            id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, category TEXT, block TEXT, place TEXT,
            status TEXT DEFAULT 'New', first_ts REAL, recurring INT DEFAULT 0, priority TEXT,
            score INT, factors TEXT, summary TEXT, team TEXT);
        CREATE TABLE IF NOT EXISTS complaints(
            id INTEGER PRIMARY KEY AUTOINCREMENT, issue_id INT, text TEXT, sim REAL, ts REAL);
        """)
