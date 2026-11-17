import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import pytz

from config import DATA_DIR, DB_FILE, DB_TIMEOUT_SECONDS, TIMEZONE, UNKNOWN_NAME
from security import generate_session_token, hash_password, tokens_match

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ schema

_SCHEMA = """
CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_key TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    session_token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (organization_id)
        REFERENCES organizations(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    attendance_date TEXT NOT NULL,
    attendance_time TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (organization_id)
        REFERENCES organizations(id)
        ON DELETE CASCADE,
    UNIQUE (organization_id, name, attendance_date)
);

CREATE INDEX IF NOT EXISTS idx_faces_org
ON faces(organization_id);

CREATE INDEX IF NOT EXISTS idx_attendance_org
ON attendance(organization_id);

CREATE INDEX IF NOT EXISTS idx_attendance_date
ON attendance(organization_id, attendance_date);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection; commit on success, roll back on error, always close."""
    conn = sqlite3.connect(
        str(DB_FILE),
        timeout=DB_TIMEOUT_SECONDS,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Create the data directory and tables. Existing data is preserved."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(_SCHEMA)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------- organizations


class OrganizationExistsError(Exception):
    """Raised when creating an organization whose key is already taken."""

@dataclass(frozen=True)
class Organization:
    id: int
    org_key: str
    password_hash: str
    session_token: str


def get_organization(org_key: str) -> Optional[Organization]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, org_key, password_hash, session_token
            FROM organizations
            WHERE org_key = ?
            """,
            (org_key,),
        ).fetchone()
    if row is None:
        return None
    return Organization(
        id=row["id"],
        org_key=row["org_key"],
        password_hash=row["password_hash"],
        session_token=row["session_token"],
    )


def create_organization(org_key: str, password: str) -> Organization:
    password_hash = hash_password(password)
    session_token = generate_session_token()
    now = _utc_now_iso()

    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO organizations (
                    org_key, password_hash, session_token, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (org_key, password_hash, session_token, now, now),
            )
            org_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise OrganizationExistsError(org_key) from exc

    return Organization(
        id=org_id,
        org_key=org_key,
        password_hash=password_hash,
        session_token=session_token,
    )


def authenticate_with_token(org_key: str, token: str) -> Optional[Organization]:
    organization = get_organization(org_key)
    if organization is None:
        return None
    if not tokens_match(organization.session_token, token):
        return None
    return organization


def reset_password(org_key: str, new_password: str) -> str:
    """Set a new password, rotate the session token and return the new token."""
    new_hash = hash_password(new_password)
    new_token = generate_session_token()

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE organizations
            SET password_hash = ?,
                session_token = ?,
                updated_at = ?
            WHERE org_key = ?
            """,
            (new_hash, new_token, _utc_now_iso(), org_key),
        )
    return new_token


# ------------------------------------------------------------------- faces


@dataclass(frozen=True, eq=False)
class FaceRecord:
    id: int
    name: str
    encoding: np.ndarray


def list_faces(org_id: int) -> List[FaceRecord]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, encoding, embedding_dimension
            FROM faces
            WHERE organization_id = ?
            ORDER BY id ASC
            """,
            (org_id,),
        ).fetchall()

    records: List[FaceRecord] = []
    for row in rows:
        try:
            encoding = np.frombuffer(
                row["encoding"],
                dtype=np.float32,
                count=row["embedding_dimension"],
            ).copy()
        except Exception:
            logger.warning("Skipping unreadable face record id=%s", row["id"])
            continue
        records.append(FaceRecord(id=row["id"], name=row["name"], encoding=encoding))
    return records


def add_face(org_id: int, name: str, encoding: np.ndarray) -> None:
    array = np.asarray(encoding, dtype=np.float32)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO faces (
                organization_id, name, encoding, embedding_dimension, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (org_id, name, array.tobytes(), int(array.size), _utc_now_iso()),
        )


def delete_face(org_id: int, face_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM faces WHERE id = ? AND organization_id = ?",
            (face_id, org_id),
        )


# -------------------------------------------------------------- attendance

_LOCAL_TZ = pytz.timezone(TIMEZONE)
_ATTENDANCE_COLUMNS = ["Name", "Time", "Date"]


def mark_attendance(org_id: int, name: str) -> bool:
    """Record attendance once per person per day.

    Returns True if a new record was created, False otherwise.
    """
    if not name or name == UNKNOWN_NAME:
        return False

    now = datetime.now(_LOCAL_TZ)
    date_str = now.strftime("%Y-%m-%d")

    with get_connection() as conn:
        already_marked = conn.execute(
            """
            SELECT 1
            FROM attendance
            WHERE organization_id = ? AND name = ? AND attendance_date = ?
            """,
            (org_id, name, date_str),
        ).fetchone()
        if already_marked:
            return False

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO attendance (
                organization_id, name, attendance_date, attendance_time, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (org_id, name, date_str, now.strftime("%H:%M:%S"), now.isoformat()),
        )
        return cursor.rowcount > 0


def load_attendance(org_id: int) -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                name AS Name,
                attendance_time AS Time,
                attendance_date AS Date
            FROM attendance
            WHERE organization_id = ?
            ORDER BY attendance_date DESC, attendance_time DESC
            """,
            (org_id,),
        ).fetchall()

    return pd.DataFrame([dict(row) for row in rows], columns=_ATTENDANCE_COLUMNS)


def clear_attendance(org_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM attendance WHERE organization_id = ?",
            (org_id,),
        )
