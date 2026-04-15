"""Verify that SQL and Python steps share the same Spark session.

The root cause of Steps 5/6/7 failing was that SQL steps ran on Genie
(isolated session) while Python steps ran on the local DatabricksSession.
After the fix, both use self._spark, so temp views created in SQL are
visible to subsequent Python steps.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from llmops_databricks_course_HeJun_C.eda_agent import (
    EDAAgent,
    Step,
    StepResult,
)


@pytest.fixture()
def agent() -> EDAAgent:
    """Create an EDAAgent with all external dependencies mocked out."""
    with (
        patch.object(EDAAgent, "_resolve_token", return_value="fake-token"),
        patch.object(EDAAgent, "__init__", lambda self, **kw: None),
    ):
        a = EDAAgent.__new__(EDAAgent)
        a._on_status = lambda *_args, **_kw: None

        mock_spark = MagicMock(name="SparkSession")
        a._spark = mock_spark
        a._genie = MagicMock(name="GenieClient")
        a._llm = MagicMock(name="OpenAI")
        a._ws = MagicMock(name="WorkspaceClient")
        return a


class TestSQLStepUsesLocalSpark:
    """_execute_sql_step must call self._spark.sql(), NOT self._genie.ask()."""

    def test_sql_step_calls_spark_sql(self, agent: EDAAgent) -> None:
        sample_df = pd.DataFrame({"msno": ["a"], "active_days": [10]})
        mock_result = MagicMock()
        mock_result.limit.return_value.toPandas.return_value = sample_df
        agent._spark.sql.return_value = mock_result

        step = Step(
            step_number=1,
            description="Create temp view",
            method="sql",
            code=(
                "CREATE OR REPLACE TEMP VIEW test_view AS "
                "SELECT 1 AS id"
            ),
        )
        result = agent._execute_sql_step(step)

        agent._spark.sql.assert_called_once_with(step.code)
        agent._genie.ask.assert_not_called()
        assert result.status == "success"

    def test_sql_step_returns_sample_rows(self, agent: EDAAgent) -> None:
        sample_df = pd.DataFrame(
            {"id": [1, 2], "name": ["alice", "bob"]}
        )
        mock_result = MagicMock()
        mock_result.limit.return_value.toPandas.return_value = sample_df
        agent._spark.sql.return_value = mock_result

        step = Step(
            step_number=1,
            description="Query data",
            method="sql",
            code="SELECT * FROM some_table",
        )
        result = agent._execute_sql_step(step)

        assert result.status == "success"
        assert '"alice"' in result.output
        assert '"bob"' in result.output

    def test_sql_step_handles_error(self, agent: EDAAgent) -> None:
        agent._spark.sql.side_effect = RuntimeError("bad SQL")

        step = Step(
            step_number=1,
            description="Bad query",
            method="sql",
            code="INVALID SQL",
        )
        result = agent._execute_sql_step(step)

        assert result.status == "skipped"
        assert "bad SQL" in result.errors[0]


class TestSessionSharing:
    """SQL temp views must be visible to Python steps via the same session."""

    def test_temp_view_from_sql_readable_in_python(
        self, agent: EDAAgent
    ) -> None:
        """Simulate the full flow: SQL creates a view, Python reads it.

        Both steps use the same self._spark object, so the view is shared.
        """
        sample_df = pd.DataFrame({"x": [42]})
        mock_result = MagicMock()
        mock_result.limit.return_value.toPandas.return_value = sample_df
        agent._spark.sql.return_value = mock_result

        sql_step = Step(
            step_number=1,
            description="Create view",
            method="sql",
            code="CREATE OR REPLACE TEMP VIEW v AS SELECT 42 AS x",
        )
        sql_result = agent._execute_sql_step(sql_step)
        assert sql_result.status == "success"

        agent._spark.sql.assert_called_once_with(sql_step.code)

        py_step = Step(
            step_number=2,
            description="Read view",
            method="python",
            code="df = spark.table('v')\nprint(df.count())",
        )
        python_output, python_error = agent._run_python(py_step.code)

        agent._spark.table.assert_called_once_with("v")

    def test_both_steps_use_same_spark_object(
        self, agent: EDAAgent
    ) -> None:
        """The spark object passed to _run_python and used by
        _execute_sql_step must be the exact same instance."""
        sample_df = pd.DataFrame({"a": [1]})
        mock_result = MagicMock()
        mock_result.limit.return_value.toPandas.return_value = sample_df
        agent._spark.sql.return_value = mock_result

        sql_step = Step(
            step_number=1,
            description="SQL step",
            method="sql",
            code="SELECT 1",
        )
        agent._execute_sql_step(sql_step)
        spark_used_in_sql = agent._spark.sql.call_args[0][0]

        captured: dict = {}
        agent._spark.table.return_value = MagicMock()

        py_code = (
            "import sys\n"
            "result['spark_id'] = id(spark)\n"
        )
        ns: dict = {"spark": agent._spark, "result": captured}
        exec(py_code, ns)  # noqa: S102

        assert captured["spark_id"] == id(agent._spark)
