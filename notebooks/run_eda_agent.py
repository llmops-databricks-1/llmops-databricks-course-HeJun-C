# Databricks notebook source
"""
Interactive EDA Agent CLI.

Run from terminal:  uv run python notebooks/run_eda_agent.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import textwrap
import threading
import time
from typing import Any

# ── path setup ───────────────────────────────────────────────────────
notebook_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.normpath(os.path.join(notebook_dir, "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# COMMAND ----------

# ── ANSI helpers ─────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE = "\033[97m"
_CLEAR_LINE = "\033[2K\r"

_TERM_WIDTH = min(shutil.get_terminal_size().columns, 80)


def _hline(char: str = "─", width: int = 0) -> str:
    w = width or _TERM_WIDTH
    return f"{_DIM}{char * w}{_RESET}"


# ── Spinner ──────────────────────────────────────────────────────────


class _Spinner:
    """Ephemeral progress indicator that overwrites itself in place."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        self._msg = ""
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def update(self, msg: str) -> None:
        with self._lock:
            self._msg = msg

    def start(self, msg: str = "") -> None:
        self._msg = msg
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write(_CLEAR_LINE)
        sys.stdout.flush()

    def _spin(self) -> None:
        idx = 0
        while self._running:
            frame = self._FRAMES[idx % len(self._FRAMES)]
            with self._lock:
                msg = self._msg
            text = f"  {_DIM}{frame} {msg}{_RESET}"
            max_w = _TERM_WIDTH - 2
            if len(msg) + 4 > max_w:
                text = text[: max_w + len(_DIM) + len(_RESET)] + _RESET
            sys.stdout.write(_CLEAR_LINE + text)
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)


# ── Status callback ─────────────────────────────────────────────────

_spinner = _Spinner()

_EPHEMERAL_MESSAGES: dict[str, str] = {
    "init": "Initializing...",
    "classify": "Classifying question...",
    "genie": "Querying database via Genie...",
    "format": "Formatting answer...",
    "metadata": "Retrieving dataset knowledge...",
    "skills": "Loading analysis skills...",
    "plan": "Creating analysis plan...",
    "report": "Generating final report...",
}


def _permanent(icon: str, text: str, color: str = _DIM) -> None:
    """Print a line that stays visible."""
    _spinner.stop()
    print(f"  {color}{icon}{_RESET} {text}")


def on_status(
    event: str, message: str, data: dict[str, Any]
) -> None:
    """Render agent events — ephemeral spinners for processing,
    permanent lines for milestones."""

    # ── Ephemeral: show spinner for background work ──
    if event in _EPHEMERAL_MESSAGES:
        _spinner.update(_EPHEMERAL_MESSAGES[event])
        if not _spinner._running:
            _spinner.start(_EPHEMERAL_MESSAGES[event])
        return

    # ── Permanent milestones ──
    if event == "classify_done":
        q_type = data.get("type", "?")
        label = (
            f"{_CYAN}Direct (SQL via Genie){_RESET}"
            if q_type == "direct"
            else f"{_MAGENTA}Exploratory (multi-step){_RESET}"
        )
        _permanent("✓", f"Question type: {label}", _GREEN)

    elif event == "genie_done":
        row_count = data.get("row_count", "?")
        _permanent(
            "✓", f"Genie returned {row_count} rows", _GREEN
        )

    elif event == "metadata_done":
        _permanent("✓", "Dataset knowledge retrieved", _GREEN)

    elif event == "skills_done":
        _permanent("✓", "Analysis skills loaded", _GREEN)

    elif event == "plan_done":
        steps = data.get("steps", [])
        _permanent(
            "✓",
            f"Analysis plan created ({len(steps)} steps)",
            _GREEN,
        )
        for s in steps:
            badge = (
                f"{_BLUE}SQL{_RESET}"
                if s["method"] == "sql"
                else f"{_MAGENTA}PY{_RESET}"
            )
            print(
                f"    {_DIM}{s['n']}.{_RESET} "
                f"[{badge}] {s['desc']}"
            )

    elif event == "step_start":
        step_n = data["step"]
        total = data["total"]
        desc = data["description"]
        method = data["method"]
        badge = "SQL" if method == "sql" else "Python"
        _spinner.update(
            f"Step {step_n}/{total}: {desc} [{badge}]"
        )
        if not _spinner._running:
            _spinner.start()

    elif event == "step_done":
        step_n = data.get("step", "?")
        if data["status"] == "success":
            _permanent(
                "✓",
                f"Step {step_n} completed",
                _GREEN,
            )
        else:
            _permanent(
                "!",
                f"Step {step_n} skipped",
                _YELLOW,
            )

    elif event == "done":
        _spinner.stop()
        elapsed = data.get("elapsed", "?")
        trace_path = data.get("trace_path", "")
        _permanent("✓", f"Finished in {elapsed}s", _GREEN)
        if trace_path:
            print(
                f"    {_DIM}Trace: {trace_path}{_RESET}"
            )


# ── Log suppression ─────────────────────────────────────────────────


def _suppress_noisy_logs() -> None:
    """Silence loguru, alembic, mlflow, and other verbose loggers
    so only the clean UI is visible in the terminal."""
    from loguru import logger as _loguru

    _loguru.remove()
    for name in (
        "alembic",
        "mlflow",
        "databricks",
        "py4j",
        "pyspark",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import random
    from datetime import datetime as _dt

    import mlflow

    mlflow.set_experiment("/Users/chenheju/eda_agent")
    _suppress_noisy_logs()

    print(f"\n{_BOLD}{_CYAN}")
    print("  ╔══════════════════════════════════════╗")
    print("  ║         EDA Agent  ·  v0.1           ║")
    print("  ╚══════════════════════════════════════╝")
    print(_RESET)

    _spinner.start("Initializing agent...")
    from llmops_databricks_course_HeJun_C.eda_agent import EDAAgent

    lakebase_host = os.environ.get(
        "LAKEBASE_HOST",
        "ep-soft-resonance-d8vba8gc.database.us-east-2.cloud.databricks.com",
    )
    lakebase_instance = os.environ.get("LAKEBASE_INSTANCE", "llmops")
    agent = EDAAgent(
        on_status=on_status,
        lakebase_host=lakebase_host,
        lakebase_instance=lakebase_instance,
    )
    _spinner.stop()
    _permanent("✓", "Agent ready", _GREEN)
    if agent.memory:
        _permanent("✓", "Lakebase memory connected", _GREEN)
    print()

    ts = _dt.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"eda-{ts}-{random.randint(100_000, 999_999)}"
    conversation_history: list[dict[str, str]] = []

    if agent.memory:
        prior = agent.load_memory(session_id)
        if prior:
            conversation_history.extend(prior)
            _permanent(
                "↻",
                f"Restored {len(prior)} messages from memory",
                _CYAN,
            )

    turn_count = 0
    while True:
        print(_hline())
        try:
            if turn_count == 0:
                prompt_text = (
                    f"  {_BOLD}{_WHITE}Ask a question"
                    f"{_RESET} "
                    f"{_DIM}(or 'quit' to exit){_RESET}\n"
                )
            else:
                prompt_text = (
                    f"  {_BOLD}{_WHITE}Follow up or ask"
                    f" a new question{_RESET} "
                    f"{_DIM}(or 'quit' to exit){_RESET}\n"
                )
            user_input = input(
                prompt_text
                + f"  {_BOLD}{_GREEN}❯{_RESET} "
            )
        except (EOFError, KeyboardInterrupt):
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        print()
        _permanent("?", f"{_BOLD}{user_input}{_RESET}", _CYAN)
        print()

        try:
            answer = agent.run(
                user_input,
                history=conversation_history or None,
                session_id=session_id,
            )
        except Exception as exc:
            _spinner.stop()
            print(f"\n  {_RED}✗ Error: {exc}{_RESET}\n")
            continue

        conversation_history.append(
            {"role": "user", "content": user_input}
        )
        conversation_history.append(
            {"role": "assistant", "content": answer}
        )
        turn_count += 1

        print(f"\n{_hline('━')}")
        print(f"{_BOLD}{_GREEN}  ◆ Answer{_RESET}")
        print(_hline("━"))
        wrapped = textwrap.indent(answer.strip(), "  ")
        print(wrapped)
        print(_hline("━"))
        print()

    _spinner.stop()
    print(
        f"\n  {_DIM}Session: {session_id}  "
        f"({turn_count} turn{'s' if turn_count != 1 else ''}){_RESET}"
    )
    print(f"  {_DIM}Goodbye!{_RESET}\n")


if __name__ == "__main__":
    main()
