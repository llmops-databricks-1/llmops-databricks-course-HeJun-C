"""EDA Agent — LLM-powered exploratory data analysis on Databricks.

Handles two types of questions:
  1. **Direct** (statistics / calculations) → Genie SQL → LLM formats answer.
  2. **Exploratory** (broader analysis) → metadata + skills → LLM plans
     multi-step analysis → execute each step (SQL or Python) → LLM report.

All calls are traced via MLflow *and* written to a local ``tracing/`` folder.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.openai
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from llmops_databricks_course_HeJun_C.genie_client import GenieClient
from llmops_databricks_course_HeJun_C.memory import LakebaseMemory

# ── Configuration ────────────────────────────────────────────────────
GENIE_SPACE_ID = "01f12d5f438713899fcb6ee851206636"
LLM_BASE_URL = (
    "https://dbc-b1b2f91a-d102.cloud.databricks.com/serving-endpoints"
)
LLM_MODEL = "course_LLM"
CATALOG = "mlops_dev"
SCHEMA = "chenheju"
VS_ENDPOINT = "llmops_course_vs_endpoint"
VS_INDEX = f"{CATALOG}.{SCHEMA}.metadata_knowledge_index"
DATABRICKS_PROFILE = "llmops-course"
MAX_PYTHON_RETRIES = 3

SKILLS_DIR = Path(__file__).parent / "skills"
TRACING_DIR = Path(__file__).resolve().parents[2] / "tracing"

mlflow.openai.autolog()

# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class Step:
    step_number: int
    description: str
    method: Literal["sql", "python"]
    code: str


@dataclass
class StepResult:
    step_number: int
    description: str
    method: str
    code: str
    status: Literal["success", "skipped"]
    output: str = ""
    errors: list[str] = field(default_factory=list)


# ── Agent ────────────────────────────────────────────────────────────


StatusCallback = Callable[[str, str, dict[str, Any]], None]


def _noop_callback(
    event: str, message: str, data: dict[str, Any]
) -> None:
    pass


class EDAAgent:
    """Orchestrates the full EDA workflow."""

    def __init__(
        self,
        on_status: StatusCallback | None = None,
        lakebase_host: str | None = None,
        lakebase_instance: str | None = None,
    ) -> None:
        self._on_status = on_status or _noop_callback
        self._emit("init", "Resolving Databricks credentials...")
        token = self._resolve_token()
        self._llm = OpenAI(base_url=LLM_BASE_URL, api_key=token)
        self._genie = GenieClient(
            GENIE_SPACE_ID, profile=DATABRICKS_PROFILE
        )
        self._ws = WorkspaceClient(profile=DATABRICKS_PROFILE)
        self._emit("init", "Connecting to Databricks (Spark)...")
        self._spark = (
            DatabricksSession.builder.profile(DATABRICKS_PROFILE)
            .serverless(True)
            .getOrCreate()
        )

        self.memory: LakebaseMemory | None = None
        if lakebase_host and lakebase_instance:
            self._emit("init", "Connecting to Lakebase memory...")
            self.memory = LakebaseMemory(
                host=lakebase_host,
                instance_name=lakebase_instance,
            )

        self._emit("init", "Agent ready.")

    def _emit(
        self,
        event: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._on_status(event, message, data or {})

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    @mlflow.trace(name="eda_agent_run")
    def run(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Answer *question* and return the response string.

        Parameters
        ----------
        history:
            Prior conversation turns as ``[{"role": ..., "content": ...}]``.
            Passed into every LLM call so the model has conversational
            context (e.g. follow-up questions).
        session_id:
            Optional session identifier. When provided, the trace is
            tagged with ``mlflow.trace.session`` and, if Lakebase memory
            is configured, messages are persisted for future retrieval.
        """
        start = time.monotonic()
        logger.info("Question: {!r}", question)

        if session_id:
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": session_id},
            )

        self._emit("classify", "Classifying question...")
        q_type = self.classify_question(question, history=history)
        self._emit(
            "classify_done",
            f"Classified as: {q_type}",
            {"type": q_type},
        )

        trace_data: dict[str, Any] = {
            "question": question,
            "classification": q_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
        }

        if q_type == "direct":
            answer = self.handle_direct_question(
                question, trace_data, history=history
            )
        else:
            answer = self.handle_exploratory_question(
                question, trace_data, history=history
            )

        elapsed = round(time.monotonic() - start, 2)
        trace_data["final_answer"] = answer
        trace_data["elapsed_seconds"] = elapsed
        trace_path = _save_local_trace(trace_data)
        self._emit(
            "done",
            f"Completed in {elapsed}s",
            {"elapsed": elapsed, "trace_path": str(trace_path)},
        )

        if session_id and self.memory:
            self.save_memory(
                session_id,
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            )

        return answer

    # ------------------------------------------------------------------
    # Memory helpers (Lakebase)
    # ------------------------------------------------------------------

    @mlflow.trace(span_type="RETRIEVER", name="memory_load")
    def load_memory(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Load previous messages from Lakebase memory."""
        if self.memory:
            return self.memory.load_messages(session_id)
        return []

    @mlflow.trace(span_type="CHAIN", name="memory_save")
    def save_memory(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Save new messages to Lakebase memory."""
        if self.memory:
            self.memory.save_messages(session_id, messages)

    # ------------------------------------------------------------------
    # Question classifier
    # ------------------------------------------------------------------

    @mlflow.trace(name="classify_question")
    def classify_question(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> Literal["direct", "exploratory"]:
        resp = self._chat(
            system=(
                "You are a question classifier. The user will give you a "
                "data analysis question. Classify it as exactly one of:\n"
                '  "direct" — a simple statistic, count, min, max, '
                "average, or lookup that can be answered with a single "
                "SQL query.\n"
                '  "exploratory" — a broader analytical or investigative '
                "question that requires multiple steps.\n"
                "Reply with ONLY the single word: direct or exploratory."
            ),
            user=question,
            history=history,
        )
        label = resp.strip().lower().strip('"')
        if label not in ("direct", "exploratory"):
            logger.warning(
                "Unexpected classifier output {!r}, defaulting to "
                "exploratory",
                label,
            )
            label = "exploratory"
        return label  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Type 1 — Direct question
    # ------------------------------------------------------------------

    @mlflow.trace(name="handle_direct_question")
    def handle_direct_question(
        self,
        question: str,
        trace_data: dict[str, Any],
        *,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self._emit("genie", "Sending question to Genie...")
        genie_result = self._genie.ask(question)
        trace_data["genie_sql"] = genie_result.sql
        trace_data["genie_rows"] = genie_result.rows[:50]
        trace_data["genie_status"] = genie_result.status

        if genie_result.status != "COMPLETED":
            return (
                "Sorry, I could not retrieve an answer from the "
                f"database. (Genie status: {genie_result.status})"
            )

        self._emit(
            "genie_done",
            f"Genie returned {len(genie_result.rows)} rows",
            {"sql": genie_result.sql, "row_count": len(genie_result.rows)},
        )
        self._emit("format", "Formatting answer...")
        answer = self._chat(
            system=(
                "You are a data analyst. The user asked a question and "
                "a SQL query was run against the database. Present the "
                "result as a concise, human-friendly answer. Include "
                "the key numbers. If the result set is a table, format "
                "it as a markdown table."
            ),
            user=(
                f"Question: {question}\n\n"
                f"SQL:\n```sql\n{genie_result.sql}\n```\n\n"
                f"Result ({len(genie_result.rows)} rows):\n"
                f"{json.dumps(genie_result.rows[:30], indent=2)}"
            ),
            history=history,
        )
        return answer

    # ------------------------------------------------------------------
    # Type 2 — Exploratory question
    # ------------------------------------------------------------------

    @mlflow.trace(name="handle_exploratory_question")
    def handle_exploratory_question(
        self,
        question: str,
        trace_data: dict[str, Any],
        *,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        self._emit("metadata", "Retrieving dataset metadata...")
        metadata = self.retrieve_metadata(question)
        self._emit(
            "metadata_done",
            f"Retrieved {len(metadata)} chars of metadata",
            {"metadata": metadata[:500]},
        )

        self._emit("skills", "Loading analysis skills...")
        skills = self.load_skills()
        self._emit(
            "skills_done",
            f"Loaded {len(skills)} chars of skill guidance",
            {"skills": skills[:300]},
        )
        trace_data["metadata_context"] = metadata
        trace_data["skills_context"] = skills

        self._emit("plan", "Creating analysis plan...")
        plan = self.create_analysis_plan(
            question, metadata, skills, history=history
        )
        self._emit(
            "plan_done",
            f"Plan created with {len(plan)} steps",
            {"steps": [
                {"n": s.step_number, "desc": s.description, "method": s.method}
                for s in plan
            ]},
        )
        trace_data["plan"] = [asdict(s) for s in plan]

        step_results: list[StepResult] = []
        for step in plan:
            self._emit(
                "step_start",
                f"Step {step.step_number}/{len(plan)}: "
                f"{step.description}",
                {"step": step.step_number, "total": len(plan),
                 "method": step.method, "description": step.description},
            )
            result = self.execute_step(step)
            self._emit(
                "step_done",
                f"Step {step.step_number}: {result.status}",
                {"step": step.step_number, "status": result.status,
                 "output_len": len(result.output),
                 "errors": result.errors},
            )
            step_results.append(result)

        trace_data["step_results"] = [asdict(r) for r in step_results]

        self._emit("report", "Generating final report...")
        report = self.generate_report(
            question, step_results, history=history
        )
        trace_data["report"] = report

        report_path = _save_report(question, report)
        trace_data["report_path"] = str(report_path)

        return report

    # ------------------------------------------------------------------
    # Metadata & skill retrieval
    # ------------------------------------------------------------------

    @mlflow.trace(name="retrieve_metadata")
    def retrieve_metadata(self, question: str) -> str:
        try:
            results = self._ws.vector_search_indexes.query_index(
                index_name=VS_INDEX,
                columns=["id", "section", "title", "content"],
                query_text=question,
                num_results=6,
            )
            chunks: list[str] = []
            for row in results.result.data_array:
                chunks.append(
                    f"[{row[0]}] {row[2]}\n{row[3]}"
                )
            return "\n\n---\n\n".join(chunks)
        except Exception:
            logger.opt(exception=True).warning(
                "Vector Search query failed, falling back to empty "
                "metadata"
            )
            return ""

    @mlflow.trace(name="load_skills")
    def load_skills(self) -> str:
        parts: list[str] = []
        if SKILLS_DIR.is_dir():
            for md in sorted(SKILLS_DIR.glob("*.md")):
                parts.append(md.read_text())
        return "\n\n---\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Analysis planning
    # ------------------------------------------------------------------

    @mlflow.trace(name="create_analysis_plan")
    def create_analysis_plan(
        self,
        question: str,
        metadata: str,
        skills: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> list[Step]:
        system_prompt = textwrap.dedent(f"""\
            You are a senior data analyst planning an exploratory data
            analysis on a Databricks lakehouse.

            ## Dataset metadata
            {metadata}

            ## Domain skill / guidelines
            {skills}

            ## Rules
            - When joining tables, ALWAYS use INNER JOIN to keep only
              IDs that exist in all relevant tables.
            - Aggregate event-level tables (user_logs, transactions) to
              user level BEFORE joining with churn labels.
            - The catalog is "{CATALOG}" and the schema is "{SCHEMA}".
              Always use fully qualified table names
              (e.g. {CATALOG}.{SCHEMA}.members).
            - SQL steps and Python steps share the SAME Spark session.
              Temp views created in SQL steps (CREATE OR REPLACE TEMP VIEW)
              are directly accessible in later Python steps via
              spark.table("view_name").
            - For Python steps, use PySpark via a variable called
              `spark` (a live DatabricksSession). Common imports
              (pyspark.sql.functions, pandas) are available.
            - Keep the plan to at most 8 steps.

            ## Output format
            Return a JSON array of step objects. Each object has:
              "step_number": int,
              "description": str,
              "method": "sql" or "python",
              "code": str  (the SQL query or Python code to execute)

            Return ONLY valid JSON, no markdown fences.
        """)

        raw = self._chat(
            system=system_prompt, user=question, history=history
        )
        raw = _strip_json_fences(raw)
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("LLM returned invalid plan JSON:\n{}", raw)
            return []

        steps: list[Step] = []
        for item in items:
            steps.append(
                Step(
                    step_number=item["step_number"],
                    description=item["description"],
                    method=item["method"],
                    code=item["code"],
                )
            )
        return steps

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    @mlflow.trace(name="execute_step")
    def execute_step(self, step: Step) -> StepResult:
        logger.info(
            "Step {} [{}]: {}",
            step.step_number,
            step.method,
            step.description,
        )

        if step.method == "sql":
            return self._execute_sql_step(step)
        return self._execute_python_step(step)

    def _execute_sql_step(self, step: Step) -> StepResult:
        """Run SQL via the local Spark session so temp views are shared
        with subsequent Python steps."""
        try:
            result_df = self._spark.sql(step.code)
            sample = result_df.limit(50).toPandas()
            rows = sample.to_dict(orient="records")
            output = json.dumps(rows, indent=2, default=str)
            return StepResult(
                step_number=step.step_number,
                description=step.description,
                method="sql",
                code=step.code,
                status="success",
                output=output,
            )
        except Exception as exc:
            return StepResult(
                step_number=step.step_number,
                description=step.description,
                method="sql",
                code=step.code,
                status="skipped",
                errors=[str(exc)],
            )

    def _execute_python_step(self, step: Step) -> StepResult:
        code = step.code
        errors: list[str] = []

        for attempt in range(1, MAX_PYTHON_RETRIES + 1):
            logger.debug(
                "Python attempt {}/{} for step {}",
                attempt,
                MAX_PYTHON_RETRIES,
                step.step_number,
            )
            output, error = self._run_python(code)
            if error is None:
                return StepResult(
                    step_number=step.step_number,
                    description=step.description,
                    method="python",
                    code=code,
                    status="success",
                    output=output,
                )
            errors.append(error)
            if attempt < MAX_PYTHON_RETRIES:
                code = self._revise_python(code, error, step)

        logger.warning(
            "Step {} skipped after {} failures",
            step.step_number,
            MAX_PYTHON_RETRIES,
        )
        return StepResult(
            step_number=step.step_number,
            description=step.description,
            method="python",
            code=code,
            status="skipped",
            errors=errors,
        )

    def _run_python(self, code: str) -> tuple[str, str | None]:
        """Execute *code* locally and capture printed output."""
        import io
        import contextlib
        import pyspark.sql.functions as F  # noqa: N812

        buf = io.StringIO()
        ns: dict[str, Any] = {
            "spark": self._spark,
            "F": F,
            "col": F.col,
            "lit": F.lit,
            "when": F.when,
            "sum": F.sum,
            "avg": F.avg,
            "count": F.count,
            "CATALOG": CATALOG,
            "SCHEMA": SCHEMA,
        }
        try:
            import pandas  # noqa: F811

            ns["pd"] = pandas
        except ImportError:
            pass

        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)  # noqa: S102
            return buf.getvalue(), None
        except Exception:
            return buf.getvalue(), traceback.format_exc()

    @mlflow.trace(name="revise_python")
    def _revise_python(
        self, code: str, error: str, step: Step
    ) -> str:
        revised = self._chat(
            system=(
                "You are a Python debugging assistant. The user tried "
                "to run a PySpark code snippet and got an error. Fix "
                "the code and return ONLY the corrected Python code, "
                "no markdown fences or explanation."
            ),
            user=(
                f"Task: {step.description}\n\n"
                f"Code:\n{code}\n\nError:\n{error}"
            ),
        )
        return _strip_code_fences(revised)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    @mlflow.trace(name="generate_report")
    def generate_report(
        self,
        question: str,
        step_results: list[StepResult],
        *,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        results_text = ""
        for r in step_results:
            status_label = (
                "COMPLETED" if r.status == "success" else "SKIPPED"
            )
            results_text += (
                f"\n### Step {r.step_number}: {r.description}\n"
                f"Method: {r.method} | Status: {status_label}\n"
            )
            if r.output:
                results_text += f"Output:\n{r.output[:3000]}\n"
            if r.errors:
                results_text += f"Errors:\n{r.errors[-1][:500]}\n"

        report = self._chat(
            system=(
                "You are a senior data analyst. Review all the step "
                "results from an exploratory analysis and write a "
                "clear, well-structured markdown report that answers "
                "the original question.\n\n"
                "The report MUST begin with a '## TLDR' section "
                "containing 3-6 bullet points that summarize the core "
                "findings, key numbers, and direct answers to the "
                "question. This section should be self-contained so a "
                "reader can get the full picture without reading "
                "further.\n\n"
                "After the TLDR, include detailed sections with "
                "supporting evidence, notable numbers, methodology "
                "notes, and any caveats. If some steps were skipped, "
                "note what was missed."
            ),
            user=(
                f"Original question: {question}\n\n"
                f"Analysis results:\n{results_text}"
            ),
            history=history,
        )
        return report

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _chat(
        self,
        *,
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        resp = self._llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_token() -> str:
        token = os.environ.get("DATABRICKS_TOKEN")
        if token:
            return token
        from databricks.sdk.core import Config

        cfg = Config(profile=DATABRICKS_PROFILE)
        if cfg.token:
            return cfg.token
        # Handles OAuth / databricks-cli auth types: call authenticate()
        # which returns {"Authorization": "Bearer <token>"}.
        try:
            headers = cfg.authenticate()
            bearer = headers.get("Authorization", "")
            if bearer.startswith("Bearer "):
                return bearer[7:]
        except Exception:
            pass
        raise RuntimeError(
            "Cannot resolve Databricks token. Set DATABRICKS_TOKEN or "
            f"configure profile '{DATABRICKS_PROFILE}' in "
            "~/.databrickscfg."
        )


# ── Tracing helpers ──────────────────────────────────────────────────


def _save_local_trace(trace_data: dict[str, Any]) -> Path:
    TRACING_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", trace_data["question"].lower())[:40]
    path = TRACING_DIR / f"{ts}_{slug}.json"
    path.write_text(json.dumps(trace_data, indent=2, default=str))
    logger.info("Local trace saved → {}", path)
    return path


def _save_report(question: str, report: str) -> Path:
    TRACING_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower())[:40]
    path = TRACING_DIR / f"{ts}_{slug}_report.md"
    path.write_text(report)
    logger.info("Report saved → {}", path)
    return path


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()
