"""A2A client for the Excel Narrator Agent (standalone service).

This is the A2A *client* side of the spreadsheet ingestion feature: the main
RAG application knows only the Excel Narrator Agent's URL. It discovers the
agent's capabilities at runtime from its AgentCard (name, version, declared
skills) and delegates the actual spreadsheet-to-text conversion to that
separate process over the Agent2Agent (A2A) protocol. This module never
imports anything from `services.excel_agent` -- the two processes are fully
decoupled and communicate only over HTTP/JSON-RPC, which is the whole
teaching point of this exercise.

Start the agent before using this client:
    .venv/bin/python -m services.excel_agent
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
from a2a.client import A2ACardResolver, A2AClientError, ClientConfig, create_client
from a2a.helpers import (
    get_artifact_text,
    get_data_parts,
    get_message_text,
    new_raw_message,
)
from a2a.types import Role, SendMessageRequest, TaskState

logger = logging.getLogger(__name__)

DEFAULT_AGENT_URL = "http://127.0.0.1:9999"

# Duplicated from services/excel_agent/narrator.py on purpose: the client
# must never import the agent's code (zero coupling is the whole teaching
# point), so it hardcodes the same two extensions it knows the agent's
# AgentCard advertises support for.
SUPPORTED_EXTENSIONS = {".xlsx", ".csv"}

_MEDIA_TYPES = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
}

_TASK_STATE_NAMES = {value: name for name, value in TaskState.items()}

# FALLBACK ONLY. The Excel Narrator Agent returns sheet_count/sheet_names as
# a proper structured data part (see get_data_parts() below) -- that is the
# protocol-native source of truth and is always preferred when present. This
# regex exists only for talking to a hypothetical *different* narrator agent
# that implements the same AgentCard skill but returns a text-only artifact
# (no data part). It matches ExcelNarrator's deterministic per-sheet header
# paragraph, e.g.:
#   "Sheet 'Enrollments' from courses.xlsx contains 12 data rows with
#    columns: Course, Instructor, Quarter, Students Enrolled, Satisfaction."
_SHEET_HEADER_RE = re.compile(
    r"^Sheet '(?P<name>.+?)' from .+ contains \d+ data rows? with columns: .+\.$",
    re.MULTILINE,
)


def _merge_data_parts(parts, into: dict[str, Any]) -> None:
    """Merge any structured data parts found in `parts` into `into`.

    This is the protocol-native way to read structured fields an agent
    attaches alongside free text in the same artifact -- no text scraping
    required. See a2a.helpers.get_data_parts / new_data_part.
    """
    for data in get_data_parts(parts):
        if isinstance(data, dict):
            into.update(data)


class A2AExcelClientError(Exception):
    """Raised when the Excel Narrator Agent is unreachable or a task fails."""


class A2AExcelClient:
    """Delegates spreadsheet-to-text conversion to the Excel Narrator Agent.

    Mirrors `DocumentExtractor`'s public shape so `main.py` can treat PDF and
    spreadsheet extraction identically: `extract_text_from_file()` returns
    `{"text": str, "page_count": int, "metadata": dict}`.
    """

    def __init__(self, agent_url: str | None = None):
        self.agent_url = agent_url or os.environ.get(
            "A2A_EXCEL_AGENT_URL", DEFAULT_AGENT_URL
        )

    def extract_text_from_file(self, file_path: str) -> dict[str, Any]:
        """Synchronous entry point (wraps the async A2A flow with asyncio.run)."""
        return asyncio.run(self._extract_text_from_file_async(file_path))

    async def _extract_text_from_file_async(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Expected one of: "
                f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

        data = path.read_bytes()
        media_type = _MEDIA_TYPES[suffix]

        # Discover the agent's capabilities from its AgentCard over HTTP --
        # the main app never imports the agent's code, only its published
        # contract. One shared, generously-timed httpx client is used for
        # both card discovery and the message exchange: per-sheet LLM
        # summaries can each take several seconds, during which the SSE
        # stream is idle, so httpx's ~5s default timeout is too short to
        # survive a multi-sheet workbook.
        artifact_texts: list[str] = []
        sheet_metadata: dict[str, Any] = {}
        final_state: int | None = None

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client, base_url=self.agent_url
            )
            try:
                agent_card = await resolver.get_agent_card()
            # The resolver wraps transport failures in its own exception
            # hierarchy (AgentCardResolutionError <- A2AClientError), so
            # catching httpx.HTTPError alone would miss the common
            # "agent not running" case.
            except (httpx.HTTPError, A2AClientError) as e:
                raise A2AExcelClientError(
                    f"Excel Narrator Agent not reachable at {self.agent_url}. "
                    "Start it first: .venv/bin/python -m services.excel_agent"
                ) from e

            logger.info(
                f"Discovered agent '{agent_card.name}' v{agent_card.version} "
                f"with skill(s): {', '.join(s.id for s in agent_card.skills)}"
            )

            client = await create_client(
                agent=agent_card,
                client_config=ClientConfig(streaming=True, httpx_client=httpx_client),
            )

            try:
                message = new_raw_message(
                    raw=data,
                    media_type=media_type,
                    filename=path.name,
                    role=Role.ROLE_USER,
                )
                request = SendMessageRequest(message=message)

                async for chunk in client.send_message(request):
                    payload_kind = chunk.WhichOneof("payload")

                    if payload_kind == "status_update":
                        status = chunk.status_update.status
                        final_state = status.state
                        state_name = _TASK_STATE_NAMES.get(status.state, status.state)
                        text = (
                            get_message_text(status.message)
                            if status.HasField("message")
                            else ""
                        )
                        logger.info(f"[Excel Narrator Agent] {state_name}: {text}")

                    elif payload_kind == "artifact_update":
                        artifact = chunk.artifact_update.artifact
                        text = get_artifact_text(artifact)
                        if text:
                            artifact_texts.append(text)
                        _merge_data_parts(artifact.parts, sheet_metadata)

                    elif payload_kind == "task":
                        # Non-streaming fallback shape: the whole Task (status +
                        # artifacts) arrives in a single chunk.
                        final_state = chunk.task.status.state
                        for artifact in chunk.task.artifacts:
                            text = get_artifact_text(artifact)
                            if text:
                                artifact_texts.append(text)
                            _merge_data_parts(artifact.parts, sheet_metadata)

                    elif payload_kind == "message":
                        text = get_message_text(chunk.message)
                        if text:
                            artifact_texts.append(text)
            except (httpx.HTTPError, A2AClientError) as e:
                raise A2AExcelClientError(
                    f"Lost connection to Excel Narrator Agent at {self.agent_url} "
                    f"while processing {path.name}."
                ) from e
            finally:
                await client.close()

        if final_state != TaskState.TASK_STATE_COMPLETED:
            state_name = _TASK_STATE_NAMES.get(final_state, final_state)
            raise A2AExcelClientError(
                f"Excel Narrator Agent did not complete {path.name} "
                f"successfully (final state: {state_name})."
            )

        narrated_text = "\n\n".join(artifact_texts)

        if "sheet_count" in sheet_metadata:
            # Protocol-native path: the agent handed back structured fields
            # in a data part alongside the text. protobuf's Struct/Value
            # JSON mapping represents all numbers as doubles, so ints come
            # back as floats (e.g. 2.0) -- cast them back explicitly.
            sheet_count = int(sheet_metadata["sheet_count"])
            sheet_names = [str(name) for name in sheet_metadata.get("sheet_names", [])]
            names_suffix = f": {', '.join(sheet_names)}" if sheet_names else ""
            logger.info(
                f"Sheet metadata from agent data part: {sheet_count} sheet(s)"
                f"{names_suffix}"
            )
        else:
            # Fallback path -- only exercised against a text-only narrator
            # agent that doesn't send a data part (see _SHEET_HEADER_RE).
            sheet_names = [
                m.group("name") for m in _SHEET_HEADER_RE.finditer(narrated_text)
            ]
            sheet_count = len(sheet_names) or 1

            if sheet_names:
                logger.info(
                    f"No data part in response; recovered {sheet_count} "
                    f"sheet(s) from narrated text: {', '.join(sheet_names)}"
                )
            else:
                logger.warning(
                    "Could not detect per-sheet headers in the narrated text; "
                    "defaulting sheet_count to 1."
                )

        return {
            "text": narrated_text,
            "page_count": sheet_count,
            "metadata": {
                "source_file": path.name,
                "file_path": str(path.absolute()),
                "page_count": sheet_count,
                "sheet_count": sheet_count,
                "file_size_bytes": path.stat().st_size,
                "extraction_method": "a2a_excel_narrator",
                "agent_url": self.agent_url,
            },
        }
