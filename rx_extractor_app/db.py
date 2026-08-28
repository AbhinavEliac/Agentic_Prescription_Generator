"""
db.py
-----
SQLite persistence.

Two tables:
  processes  - one row per "Start Process" click: device, status, and the
               csv/xlsx paths that this run's generations are written to.
  history    - one row per query/response, linked to a process.

Why this matters for "refresh should not lose the process": Streamlit
re-runs the whole script top-to-bottom on every page refresh, wiping
st.session_state. This module is how app.py figures out, on that fresh
run, "is there still an ACTIVE process?" and reattaches to it -- reloading
its history for display instead of starting over.
"""
import os
import sqlite3
import datetime
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS processes (
    process_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    device       TEXT NOT NULL,
    model_name   TEXT,
    model_label  TEXT,
    status       TEXT NOT NULL DEFAULT 'active',   -- active | stopped
    csv_path     TEXT NOT NULL,
    xlsx_path    TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    stopped_at   TEXT
);

CREATE TABLE IF NOT EXISTS history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    process_id      INTEGER NOT NULL,
    query           TEXT NOT NULL,
    output          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    generation_time REAL,
    FOREIGN KEY (process_id) REFERENCES processes(process_id)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        cur = conn.execute("PRAGMA table_info(history)")
        cols = [r["name"] for r in cur.fetchall()]
        if "generation_time" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN generation_time REAL")
        if "audio_path" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN audio_path TEXT")

        cur_p = conn.execute("PRAGMA table_info(processes)")
        p_cols = [r["name"] for r in cur_p.fetchall()]
        if "model_name" not in p_cols:
            conn.execute("ALTER TABLE processes ADD COLUMN model_name TEXT")
        if "model_label" not in p_cols:
            conn.execute("ALTER TABLE processes ADD COLUMN model_label TEXT")


def create_process(
    name: str, device: str, csv_path: str, xlsx_path: str, model_name: str = None, model_label: str = None
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO processes (name, device, status, csv_path, xlsx_path, created_at, model_name, model_label) "
            "VALUES (?, ?, 'active', ?, ?, ?, ?, ?)",
            (name, device, csv_path, xlsx_path, datetime.datetime.now().isoformat(), model_name, model_label),
        )
        return cur.lastrowid


def get_active_process():
    """Most recent still-running process, or None. This is the reattach point."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM processes WHERE status = 'active' ORDER BY process_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def stop_process(process_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE processes SET status = 'stopped', stopped_at = ? WHERE process_id = ?",
            (datetime.datetime.now().isoformat(), process_id),
        )


def add_history(process_id: int, query: str, output: str, generation_time: float = None, audio_path: str = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO history (process_id, query, output, created_at, generation_time, audio_path) VALUES (?, ?, ?, ?, ?, ?)",
            (process_id, query, output, datetime.datetime.now().isoformat(), generation_time, audio_path),
        )



def get_history(process_id: int = None, limit: int = 50):
    with get_conn() as conn:
        if process_id is not None:
            rows = conn.execute(
                "SELECT * FROM history WHERE process_id = ? ORDER BY id DESC LIMIT ?", (process_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_process(process_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM processes WHERE process_id = ?", (process_id,)).fetchone()
        return dict(row) if row else None


def list_processes():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM processes ORDER BY process_id DESC").fetchall()
        return [dict(r) for r in rows]


def delete_process(process_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT csv_path, xlsx_path FROM processes WHERE process_id = ?", (process_id,)).fetchone()
        if row:
            for p in (row["csv_path"], row["xlsx_path"]):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        conn.execute("DELETE FROM history WHERE process_id = ?", (process_id,))
        conn.execute("DELETE FROM processes WHERE process_id = ?", (process_id,))


def delete_all_processes():
    with get_conn() as conn:
        rows = conn.execute("SELECT csv_path, xlsx_path FROM processes").fetchall()
        for row in rows:
            for p in (row["csv_path"], row["xlsx_path"]):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        conn.execute("DELETE FROM history")
        conn.execute("DELETE FROM processes")
