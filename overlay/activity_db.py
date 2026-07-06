"""SQLite activity store for the productivity sensor (PHASE-02B).

100% local. Stores 5-second aggregates of gaze zone, presence, active window
and keyboard/mouse COUNTS. It NEVER stores key content (see activity_logger
for the keylogging prohibition) - only per-interval event counts.

WAL mode from the first write so the file survives the overlay's hard exit
(os._exit): each interval is a committed transaction, so a process kill loses
at most the in-progress interval, never a committed one. On open, any session
left without an end (previous hard kill) is closed to its last sample.
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

INTERVAL_SECONDS = 5
_CHECKPOINT_EVERY = 60          # passive WAL checkpoint cadence (intervals)

# Per-sample tracking states.
STATE_OK = "ok"
STATE_FACE_LOST = "face_lost"
STATE_POSE_EXTREME = "pose_extreme"
STATE_OFFSCREEN = "offscreen"
STATE_CAMERA_LOST = "camera_lost"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    start  TEXT NOT NULL,
    end    TEXT,
    notes  TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    ts             TEXT NOT NULL,
    session_id     INTEGER,
    gaze_zone      TEXT,      -- "c<col>r<row>" (modal on-screen zone) or "off"
    on_screen      REAL,      -- fraction 0..1 of frames on-screen
    tracking_state TEXT,      -- modal state (see STATE_*)
    face_present   REAL,      -- fraction 0..1 of frames with a face
    process        TEXT,      -- modal foreground process (e.g. Code.exe)
    window_title   TEXT,      -- modal title, or NULL under --no-titles
    kb_events      INTEGER,   -- key PRESS count this interval (no content)
    mouse_events   INTEGER,   -- mouse click/scroll/throttled-move count
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id);
"""


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class ActivityDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        # WAL from the first write; NORMAL is crash/kill-safe for committed
        # transactions (only a power loss, not a process kill, risks the last
        # commit) and much faster than FULL.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._recover_open_sessions()
        self._writes = 0

    def _recover_open_sessions(self) -> None:
        """Close any session left open by a previous hard kill, dating its end
        to that session's last sample (or its start if it has none)."""
        rows = self._conn.execute(
            "SELECT id, start FROM sessions WHERE end IS NULL").fetchall()
        for sid, start in rows:
            last = self._conn.execute(
                "SELECT MAX(ts) FROM samples WHERE session_id=?",
                (sid,)).fetchone()[0]
            self._conn.execute(
                "UPDATE sessions SET end=?, notes=COALESCE(notes,'') || "
                "' [closed on recovery]' WHERE id=?", (last or start, sid))

    def start_session(self, notes: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO sessions(start, notes) VALUES(?, ?)",
            (_now_iso(), notes))
        return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        self._conn.execute("UPDATE sessions SET end=? WHERE id=?",
                           (_now_iso(), session_id))

    def insert_sample(self, session_id: int, gaze_zone: str, on_screen: float,
                      tracking_state: str, face_present: float,
                      process: str | None, window_title: str | None,
                      kb_events: int, mouse_events: int,
                      ts: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO samples(ts, session_id, gaze_zone, on_screen, "
            "tracking_state, face_present, process, window_title, kb_events, "
            "mouse_events) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ts or _now_iso(), session_id, gaze_zone, round(on_screen, 3),
             tracking_state, round(face_present, 3), process, window_title,
             int(kb_events), int(mouse_events)))
        # isolation_level=None => each statement autocommits, so a hard kill
        # never loses a written interval. Periodically fold the WAL back.
        self._writes += 1
        if self._writes % _CHECKPOINT_EVERY == 0:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def estimate_size_per_day(self) -> dict:
        """Extrapolate on-disk bytes/day from the current rows, plus an
        analytic fallback when there are too few rows yet."""
        n = self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        rows_per_day = 86400 / INTERVAL_SECONDS
        try:
            file_bytes = self.path.stat().st_size
        except OSError:
            file_bytes = 0
        bytes_per_row = (file_bytes / n) if n else 130.0  # analytic estimate
        return {
            "rows": n,
            "rows_per_day": int(rows_per_day),
            "bytes_per_row": round(bytes_per_row, 1),
            "mb_per_day": round(bytes_per_row * rows_per_day / 1e6, 2),
        }

    def query(self, sql: str, params: tuple = ()) -> list:
        return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self._conn.close()


# Sample review queries (used in the 2B gate report and reusable in 2C).
SAMPLE_QUERIES: dict[str, str] = {
    "time_per_app": """
        SELECT process,
               ROUND(COUNT(*) * :iv / 60.0, 1) AS minutes
        FROM samples
        WHERE process IS NOT NULL
        GROUP BY process
        ORDER BY minutes DESC
        LIMIT 15;
    """,
    "time_per_gaze_zone": """
        SELECT gaze_zone,
               ROUND(COUNT(*) * :iv / 60.0, 1) AS minutes
        FROM samples
        GROUP BY gaze_zone
        ORDER BY minutes DESC;
    """,
    "on_off_screen_ratio": """
        SELECT ROUND(AVG(on_screen) * 100, 1)       AS pct_on_screen,
               ROUND((1 - AVG(on_screen)) * 100, 1)  AS pct_off_screen,
               ROUND(COUNT(*) * :iv / 60.0, 1)       AS total_minutes
        FROM samples;
    """,
    "presence_gaps": """
        -- Intervals where the face was mostly absent (a presence gap),
        -- with the active process during the gap.
        SELECT ts, tracking_state, process,
               ROUND(face_present, 2) AS face_present
        FROM samples
        WHERE face_present < 0.5
        ORDER BY ts;
    """,
    "activity_per_app": """
        -- Keyboard/mouse intensity per app (events per active minute).
        SELECT process,
               SUM(kb_events)    AS keys,
               SUM(mouse_events) AS mouse,
               ROUND(COUNT(*) * :iv / 60.0, 1) AS minutes
        FROM samples
        WHERE process IS NOT NULL
        GROUP BY process
        ORDER BY (SUM(kb_events) + SUM(mouse_events)) DESC
        LIMIT 15;
    """,
}
