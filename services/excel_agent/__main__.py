"""Enables `python -m services.excel_agent` to start the Excel Narrator Agent.

Equivalent to running `services/excel_agent/server.py` directly; --host/--port
flags and the EXCEL_AGENT_HOST/EXCEL_AGENT_PORT env vars are handled by
`server.main()`'s own argument parsing.
"""

from services.excel_agent.server import main

if __name__ == "__main__":
    main()
