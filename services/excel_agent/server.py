"""A2A server for the Excel Narrator Agent (standalone service).

This process knows nothing about the main RAG application: it never imports
`config` or `src`. It receives raw spreadsheet bytes over the A2A protocol,
narrates them into natural-language text via
`services.excel_agent.narrator.ExcelNarrator`, and returns the result as a
task artifact, streaming a status update for every meaningful step along the
way (reading the file, narrating each sheet, completing). The main app talks
to this process only over HTTP/JSON-RPC, through
`src/data_pipeline/a2a_excel_client.py` -- it never imports this package.

Run:
    .venv/bin/python -m services.excel_agent
    (equivalently: .venv/bin/python services/excel_agent/server.py --port 9999)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import uvicorn
from a2a.helpers import (
    new_data_part,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from dotenv import load_dotenv
from starlette.applications import Starlette

from services.excel_agent.narrator import ExcelNarrator, NarratorError

# Picks up the main project's .env when this server is launched from the
# project root (`python -m services.excel_agent`). This is the ONLY place
# this package touches the main project: reading plain environment
# variables, never importing `config` or `src`.
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_HOST = os.environ.get("EXCEL_AGENT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("EXCEL_AGENT_PORT", "9999"))


def card_url(host: str, port: int) -> str:
    visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{visible_host}:{port}"


def build_agent_card(host: str, port: int) -> AgentCard:
    skill = AgentSkill(
        id="excel_to_narrative",
        name="Excel to Narrative",
        description=(
            "Converts an uploaded Excel workbook (.xlsx) or CSV file into "
            "descriptive natural-language passages -- one context paragraph "
            "per sheet plus one sentence per data row -- suitable for "
            "embedding and retrieval in a RAG pipeline."
        ),
        input_modes=[
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
        ],
        output_modes=["text/plain"],
        tags=["a2a", "rag", "excel"],
        examples=[
            "Narrate data/courses.xlsx for indexing",
            "Convert enrollment_data.csv into RAG-ready text",
        ],
    )
    return AgentCard(
        name="Excel Narrator Agent",
        description=(
            "Standalone A2A agent that converts spreadsheets (.xlsx, .csv) "
            "into descriptive natural-language passages for downstream RAG "
            "indexing. The main RAG app never imports this agent's code -- "
            "it discovers this AgentCard over HTTP and delegates conversion "
            "via the A2A protocol."
        ),
        version="0.1.0",
        default_input_modes=[
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
        ],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=card_url(host, port),
                protocol_version="1.0",
            )
        ],
        skills=[skill],
    )


class ExcelNarratorExecutor(AgentExecutor):
    """Bridges incoming A2A requests to `ExcelNarrator`."""

    def __init__(self) -> None:
        self.narrator = ExcelNarrator()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        raw_part = None
        for part in context.message.parts:
            if part.HasField("raw"):
                raw_part = part
                break

        if raw_part is None:
            logger.warning("Request had no raw file part attached; failing the task.")
            await updater.update_status(
                state=TaskState.TASK_STATE_FAILED,
                message=new_text_message(
                    "No file was attached to the request. The Excel Narrator "
                    "Agent expects the spreadsheet bytes as a raw Part "
                    "(see a2a.helpers.new_raw_message)."
                ),
            )
            return

        filename = raw_part.filename or "upload"
        media_type = raw_part.media_type or "application/octet-stream"
        data = raw_part.raw

        logger.info(f"Received {filename} ({media_type}, {len(data)} bytes)")
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message(f"Reading {filename}..."),
        )

        try:
            workbook = await asyncio.to_thread(self.narrator.load, data, filename)
        except NarratorError as e:
            logger.warning(f"Failed to parse {filename}: {e}")
            await updater.update_status(
                state=TaskState.TASK_STATE_FAILED,
                message=new_text_message(f"Could not read {filename}: {e}"),
            )
            return

        total_sheets = len(workbook.sheets)
        sheet_texts = []
        for index, sheet in enumerate(workbook.sheets, start=1):
            await updater.update_status(
                state=TaskState.TASK_STATE_WORKING,
                message=new_text_message(
                    f"Narrating sheet {index}/{total_sheets}: '{sheet.name}'..."
                ),
            )
            sheet_text = await asyncio.to_thread(
                self.narrator.narrate_sheet, workbook, sheet
            )
            sheet_texts.append(sheet_text)

        full_text = "\n\n".join(sheet_texts)

        # An artifact's parts can freely mix content types: here the same
        # artifact carries both the narrated text (the actual RAG payload)
        # and a structured data part with the sheet metadata the narrator
        # already computed. Callers that want sheet_count/sheet_names get
        # them natively via get_data_parts() instead of having to scrape
        # them back out of the human-readable text.
        sheet_metadata = {
            "sheet_count": total_sheets,
            "sheet_names": workbook.sheet_names,
            "row_count": workbook.row_count,
        }

        await updater.add_artifact(
            parts=[
                new_text_part(text=full_text, media_type="text/plain"),
                new_data_part(data=sheet_metadata, media_type="application/json"),
            ]
        )
        await updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message(
                f"Narrated {total_sheets} sheet(s) from {filename}."
            ),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError(
            "Cancel is not supported by the Excel Narrator Agent."
        )


def create_app(host: str, port: int) -> Starlette:
    agent_card = build_agent_card(host, port)
    request_handler = DefaultRequestHandler(
        agent_executor=ExcelNarratorExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, "/"))
    return Starlette(routes=routes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Excel Narrator Agent A2A server."
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind (default: {DEFAULT_PORT}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    url = card_url(args.host, args.port)

    print(f"Starting Excel Narrator Agent on {url}", file=sys.stderr)
    print(f"Agent card: {url}{AGENT_CARD_WELL_KNOWN_PATH}", file=sys.stderr)
    print("main app: python main.py process data/courses.xlsx", file=sys.stderr)

    app = create_app(args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
