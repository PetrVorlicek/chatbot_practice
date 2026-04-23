from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass(slots=True)
class Ticket:
    id: int
    title: str
    description: str
    status: TicketStatus
    created_at: datetime


class TicketStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        '{TicketStatus.OPEN.value}',
                        '{TicketStatus.IN_PROGRESS.value}',
                        '{TicketStatus.RESOLVED.value}',
                        '{TicketStatus.CLOSED.value}'
                    )),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tickets_status_created_at
                ON tickets(status, created_at DESC)
                """
            )
            conn.commit()

    def create_ticket(
        self,
        *,
        title: str,
        description: str,
        status: TicketStatus = TicketStatus.OPEN,
    ) -> Ticket:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO tickets (title, description, status)
                VALUES (?, ?, ?)
                RETURNING id, title, description, status, created_at
                """,
                (title, description, status.value),
            )
            row = cursor.fetchone()
            conn.commit()

        if row is None:
            raise RuntimeError("Failed to create ticket.")
        return _row_to_ticket(row)

    def get_ticket(self, ticket_id: int) -> Ticket | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, title, description, status, created_at
                FROM tickets
                WHERE id = ?
                """,
                (ticket_id,),
            ).fetchone()

        return _row_to_ticket(row) if row else None

    def list_tickets(
        self,
        *,
        status: TicketStatus | None = None,
        limit: int = 100,
    ) -> list[Ticket]:
        safe_limit = max(1, min(limit, 500))

        with sqlite3.connect(self.db_path) as conn:
            if status is None:
                rows = conn.execute(
                    """
                    SELECT id, title, description, status, created_at
                    FROM tickets
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, title, description, status, created_at
                    FROM tickets
                    WHERE status = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (status.value, safe_limit),
                ).fetchall()

        return [_row_to_ticket(row) for row in rows]

    def update_status(self, ticket_id: int, status: TicketStatus) -> Ticket | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                UPDATE tickets
                SET status = ?
                WHERE id = ?
                RETURNING id, title, description, status, created_at
                """,
                (status.value, ticket_id),
            ).fetchone()
            conn.commit()

        return _row_to_ticket(row) if row else None


def _row_to_ticket(row: tuple[object, ...]) -> Ticket:
    return Ticket(
        id=int(row[0]),
        title=str(row[1]),
        description=str(row[2]),
        status=TicketStatus(str(row[3])),
        created_at=datetime.fromisoformat(str(row[4])),
    )
