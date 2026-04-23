from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from db.ticket_store import TicketStatus, TicketStore
from settings import settings


ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True, slots=True)
class AITool:
    name: str
    schema: dict[str, Any]
    handler: ToolHandler


ticket_store = TicketStore(settings.tickets_db_path)


def create_support_ticket(args: dict[str, Any]) -> str:
    title = str(args.get("title", "")).strip()
    description = str(args.get("description", "")).strip()
    if not title:
        raise ValueError("Ticket title is required.")
    if not description:
        raise ValueError("Ticket description is required.")

    ticket = ticket_store.create_ticket(
        title=title,
        description=description,
        status=TicketStatus.OPEN,
    )
    return json.dumps(
        {
            "ticket_id": ticket.id,
            "title": ticket.title,
            "status": ticket.status.value,
            "created_at": ticket.created_at.isoformat(),
        }
    )


def add_two_numbers(args: dict[str, Any]) -> str:
    if "a" not in args or "b" not in args:
        raise ValueError("Both 'a' and 'b' are required.")

    try:
        a = float(args["a"])
        b = float(args["b"])
    except (TypeError, ValueError) as exc:
        raise ValueError("'a' and 'b' must be numbers.") from exc

    return json.dumps(
        {
            "a": a,
            "b": b,
            "result": a + b,
        }
    )


CREATE_SUPPORT_TICKET_TOOL = AITool(
    name="create_support_ticket",
    schema={
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create a support ticket when the user indicates they want to end chat "
                "without resolution or asks to escalate/handoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short summary of the issue.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description from the conversation.",
                    },
                },
                "required": ["title", "description"],
                "additionalProperties": False,
            },
        },
    },
    handler=create_support_ticket,
)

ADD_TWO_NUMBERS_TOOL = AITool(
    name="add_two_numbers",
    schema={
        "type": "function",
        "function": {
            "name": "add_two_numbers",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number."},
                    "b": {"type": "number", "description": "Second number."},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
    handler=add_two_numbers,
)


class ToolRegistry:
    def __init__(self, tools: list[AITool]) -> None:
        self._tools = tools
        self._handlers = {tool.name: tool.handler for tool in tools}

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self._tools]

    def execute(self, tool_name: str, tool_args_json: str | None) -> str:
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        parsed_args: dict[str, Any] = {}
        if tool_args_json:
            parsed_raw = json.loads(tool_args_json)
            if isinstance(parsed_raw, dict):
                parsed_args = parsed_raw
            else:
                raise ValueError("Tool arguments must be a JSON object.")

        return handler(parsed_args)


tool_registry = ToolRegistry([CREATE_SUPPORT_TICKET_TOOL, ADD_TWO_NUMBERS_TOOL])
AI_TOOLS = tool_registry.openai_tools
