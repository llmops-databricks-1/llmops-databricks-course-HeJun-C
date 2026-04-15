"""EDA Agent SQL Benchmark Evaluation.

Parses ``evaluation_sql_benchmark.md``, runs each golden query on
Databricks via Spark, sends the same question through the EDA Agent,
then uses an LLM judge to compare the two results (pass / fail).

Run:  uv run python evaluation.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from databricks.connect import DatabricksSession
from openai import OpenAI

# ── Path setup ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llmops_databricks_course_HeJun_C.eda_agent import (  # noqa: E402
    CATALOG,
    DATABRICKS_PROFILE,
    LLM_BASE_URL,
    LLM_MODEL,
    SCHEMA,
    EDAAgent,
)

BENCHMARK_PATH = Path(__file__).resolve().parent / "evaluation_sql_benchmark.md"
TRACING_DIR = Path(__file__).resolve().parent / "tracing"

TABLE_NAMES = ["train", "members", "transactions", "user_logs"]


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class BenchmarkCase:
    number: int
    question: str
    difficulty: str
    golden_sql: str


@dataclass
class EvalResult:
    number: int
    question: str
    difficulty: str
    agent_classification: str = ""
    golden_answer: str = ""
    agent_answer: str = ""
    verdict: str = ""
    judge_reason: str = ""
    elapsed_s: float = 0.0
    error: str = ""


# ── Parse benchmark markdown ────────────────────────────────────────


def parse_benchmark(path: Path) -> list[BenchmarkCase]:
    text = path.read_text()
    cases: list[BenchmarkCase] = []
    blocks = re.split(r"(?=^## \d+\.)", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block.startswith("## "):
            continue
        header_match = re.match(
            r"^## (\d+)\.\s+(.+)$", block, re.MULTILINE
        )
        if not header_match:
            continue
        number = int(header_match.group(1))
        question = header_match.group(2).strip()
        question = re.sub(r"`", "", question)

        diff_match = re.search(
            r"\*\*Difficulty:\*\*\s*(\w+)", block
        )
        difficulty = diff_match.group(1) if diff_match else "Unknown"

        sql_match = re.search(
            r"```sql\n(.*?)```", block, re.DOTALL
        )
        golden_sql = sql_match.group(1).strip() if sql_match else ""

        if golden_sql:
            cases.append(
                BenchmarkCase(
                    number=number,
                    question=question,
                    difficulty=difficulty,
                    golden_sql=golden_sql,
                )
            )
    return cases


# ── Fully-qualify table names ────────────────────────────────────────


def _qualify_sql(sql: str) -> str:
    """Prefix bare table names with catalog.schema."""
    qualified = sql
    for tbl in TABLE_NAMES:
        qualified = re.sub(
            rf"(?<![.\w`])\b{tbl}\b(?!\s*\.)",
            f"`{CATALOG}`.`{SCHEMA}`.`{tbl}`",
            qualified,
        )
    return qualified


# ── Run golden query ─────────────────────────────────────────────────


def run_golden_query(spark: Any, sql: str) -> str:  # noqa: ANN401
    qualified = _qualify_sql(sql)
    df = spark.sql(qualified)
    pdf = df.limit(100).toPandas()
    return pdf.to_string(index=False)


# ── LLM judge ───────────────────────────────────────────────────────


def judge_answer(
    llm: OpenAI,
    question: str,
    golden_result: str,
    agent_answer: str,
) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is 'pass' or 'fail'."""
    prompt = (
        "You are an evaluation judge. Compare the AGENT ANSWER against "
        "the GOLDEN RESULT to determine if the agent's answer is "
        "factually correct and consistent with the golden result.\n\n"
        "Rules:\n"
        "- The agent answer may be in natural language; the golden "
        "result is raw query output. Focus on whether the key numbers "
        "and facts match.\n"
        "- Minor rounding differences (±0.01) are acceptable.\n"
        "- The agent may include extra context or formatting — that's "
        "fine as long as the core numbers are correct.\n"
        "- If the agent says it could not answer or returns an error, "
        "that is a FAIL.\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"verdict": "pass" or "fail", "reason": "one sentence"}'
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"GOLDEN RESULT:\n{golden_result}\n\n"
        f"AGENT ANSWER:\n{agent_answer}"
    )
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj.get("verdict", "fail"), obj.get("reason", "")
    except json.JSONDecodeError:
        if "pass" in raw.lower():
            return "pass", raw
        return "fail", f"Could not parse judge response: {raw}"


# ── Status callback that captures classification ─────────────────────


class _ClassificationCapture:
    """Status callback that records the agent's question classification."""

    def __init__(self) -> None:
        self.classification: str = ""

    def __call__(
        self, event: str, message: str, data: dict[str, Any]
    ) -> None:
        if event == "classify_done" and "type" in data:
            self.classification = data["type"]


# ── Suppress noisy logs ─────────────────────────────────────────────


def _suppress_logs() -> None:
    try:
        from loguru import logger as _loguru

        _loguru.remove()
    except Exception:
        pass
    for name in (
        "alembic", "mlflow", "databricks",
        "py4j", "pyspark", "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── ANSI helpers ─────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_WHITE = "\033[97m"


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    mlflow.set_experiment("/Users/chenheju/eda_agent_eval")
    _suppress_logs()

    print(f"\n{_BOLD}{_CYAN}")
    print("  ╔══════════════════════════════════════════╗")
    print("  ║   EDA Agent — SQL Benchmark Evaluation   ║")
    print("  ╚══════════════════════════════════════════╝")
    print(_RESET)

    cases = parse_benchmark(BENCHMARK_PATH)
    print(f"  Loaded {_BOLD}{len(cases)}{_RESET} benchmark cases\n")

    capture = _ClassificationCapture()
    print(f"  {_DIM}Initializing agent...{_RESET}", flush=True)
    agent = EDAAgent(on_status=capture)
    print(f"  {_GREEN}✓{_RESET} Agent ready\n")

    print(f"  {_DIM}Connecting to Spark for golden queries...{_RESET}", flush=True)
    spark = (
        DatabricksSession.builder.profile(DATABRICKS_PROFILE)
        .serverless(True)
        .getOrCreate()
    )
    print(f"  {_GREEN}✓{_RESET} Spark ready\n")

    token = EDAAgent._resolve_token()
    llm = OpenAI(base_url=LLM_BASE_URL, api_key=token)

    results: list[EvalResult] = []

    for i, case in enumerate(cases, 1):
        tag = f"[{i}/{len(cases)}]"
        q_short = (
            case.question[:58] + ".."
            if len(case.question) > 60
            else case.question
        )
        print(
            f"  {_BOLD}{_WHITE}{tag}{_RESET} "
            f"Q{case.number} ({case.difficulty}): {q_short}"
        )

        result = EvalResult(
            number=case.number,
            question=case.question,
            difficulty=case.difficulty,
        )

        # 1) Golden query
        print(f"    {_DIM}Running golden query...{_RESET}", end="", flush=True)
        try:
            golden_answer = run_golden_query(spark, case.golden_sql)
            result.golden_answer = golden_answer
            print(f"\r    {_GREEN}✓{_RESET} Golden query done       ")
        except Exception as exc:
            result.error = f"Golden query failed: {exc}"
            result.verdict = "error"
            results.append(result)
            print(f"\r    {_RED}✗ Golden query failed: {exc}{_RESET}")
            print()
            continue

        # 2) Agent answer
        print(f"    {_DIM}Running agent...{_RESET}", end="", flush=True)
        capture.classification = ""
        start = time.monotonic()
        try:
            with mlflow.start_run(run_name=f"eval_Q{case.number}"):
                agent_answer = agent.run(case.question)
            result.agent_answer = agent_answer
            result.agent_classification = capture.classification
            result.elapsed_s = round(time.monotonic() - start, 1)
            print(
                f"\r    {_GREEN}✓{_RESET} Agent answered "
                f"[{capture.classification}] ({result.elapsed_s}s)       "
            )
        except Exception as exc:
            result.error = f"Agent failed: {exc}"
            result.verdict = "fail"
            result.elapsed_s = round(time.monotonic() - start, 1)
            results.append(result)
            print(f"\r    {_RED}✗ Agent failed: {exc}{_RESET}")
            print()
            continue

        # 3) LLM Judge
        print(f"    {_DIM}Judging...{_RESET}", end="", flush=True)
        try:
            verdict, reason = judge_answer(
                llm, case.question, golden_answer, agent_answer
            )
            result.verdict = verdict
            result.judge_reason = reason
        except Exception as exc:
            result.verdict = "error"
            result.judge_reason = f"Judge failed: {exc}"

        color = _GREEN if result.verdict == "pass" else _RED
        symbol = "✓" if result.verdict == "pass" else "✗"
        print(
            f"\r    {color}{symbol} {result.verdict.upper()}{_RESET}"
            f"  {_DIM}{result.judge_reason}{_RESET}                "
        )
        print()
        results.append(result)

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'═' * 94}")
    print(f"{_BOLD}{_CYAN}  EVALUATION RESULTS{_RESET}")
    print(f"{'═' * 94}")

    header = (
        f"  {'#':>3}  {'Diff.':<8}  {'Class.':<12}  "
        f"{'Verdict':<7}  {'Time':>6}  {'Question':<42}"
    )
    print(f"{_BOLD}{header}{_RESET}")
    print(f"  {'─' * 88}")

    for r in results:
        v_color = (
            _GREEN if r.verdict == "pass"
            else _RED if r.verdict == "fail"
            else _YELLOW
        )
        q_short = (
            r.question[:40] + ".."
            if len(r.question) > 42
            else r.question
        )
        cls = r.agent_classification or "—"
        print(
            f"  {r.number:>3}  {r.difficulty:<8}  {cls:<12}  "
            f"{v_color}{r.verdict.upper():<7}{_RESET}  "
            f"{r.elapsed_s:>5.1f}s  {q_short}"
        )

    print(f"  {'─' * 88}")

    total = len(results)
    passed = sum(1 for r in results if r.verdict == "pass")
    failed = sum(1 for r in results if r.verdict == "fail")
    errors = sum(1 for r in results if r.verdict == "error")

    print(
        f"\n  {_BOLD}Total: {total}   "
        f"{_GREEN}Pass: {passed}{_RESET}{_BOLD}   "
        f"{_RED}Fail: {failed}{_RESET}{_BOLD}   "
        f"{_YELLOW}Error: {errors}{_RESET}"
    )
    if total > 0:
        rate = passed / total * 100
        rate_color = _GREEN if rate >= 70 else _YELLOW if rate >= 50 else _RED
        print(
            f"  {_BOLD}Pass rate: "
            f"{rate_color}{rate:.1f}%{_RESET}"
        )

    print()
    for diff in ("Easy", "Medium", "Hard"):
        subset = [r for r in results if r.difficulty == diff]
        if subset:
            p = sum(1 for r in subset if r.verdict == "pass")
            pct = p / len(subset) * 100
            diff_color = _GREEN if pct >= 70 else _YELLOW if pct >= 50 else _RED
            print(
                f"    {diff:<8} "
                f"{diff_color}{p}/{len(subset)} "
                f"({pct:.0f}%){_RESET}"
            )

    # ── Log to MLflow ────────────────────────────────────────────────
    with mlflow.start_run(run_name="eval_summary"):
        mlflow.log_metric("total_questions", total)
        mlflow.log_metric("pass_count", passed)
        mlflow.log_metric("fail_count", failed)
        mlflow.log_metric("error_count", errors)
        if total > 0:
            mlflow.log_metric("pass_rate", passed / total)
        for diff in ("Easy", "Medium", "Hard"):
            subset = [r for r in results if r.difficulty == diff]
            if subset:
                p = sum(1 for r in subset if r.verdict == "pass")
                mlflow.log_metric(
                    f"pass_rate_{diff.lower()}", p / len(subset)
                )

    # ── Save results to tracing folder ───────────────────────────────
    TRACING_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = TRACING_DIR / f"{ts}_eval_results.json"
    serializable = []
    for r in results:
        serializable.append({
            "number": r.number,
            "question": r.question,
            "difficulty": r.difficulty,
            "agent_classification": r.agent_classification,
            "verdict": r.verdict,
            "judge_reason": r.judge_reason,
            "elapsed_s": r.elapsed_s,
            "error": r.error,
            "golden_answer_preview": r.golden_answer[:500],
            "agent_answer_preview": r.agent_answer[:500],
        })
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n  {_DIM}Results saved → {out_path}{_RESET}")
    print()


if __name__ == "__main__":
    main()
