# Databricks notebook source
"""Batch entrypoint for the EDA agent, deployed via DABs.

Reads a ``question`` job parameter, runs the agent once, writes the
result to a UC table, and prints it to the job log. Designed for
scheduled or on-demand Databricks jobs under a Service Principal.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

from llmops_databricks_course_HeJun_C.eda_agent import EDAAgent

# COMMAND ----------

dbutils = globals().get("dbutils")


def _get_param(name: str, default: str = "") -> str:
    """Read a widget/job parameter, fall back to env var, then default."""
    if dbutils is not None:
        try:
            dbutils.widgets.text(name, default)
            value = dbutils.widgets.get(name)
            if value:
                return value
        except Exception:
            pass
    return os.environ.get(name.upper(), default)


env = _get_param("env", "dev")
git_sha = _get_param("git_sha", "local")
run_id = _get_param("run_id", "manual")
question = _get_param(
    "question",
    "What is the overall churn rate in the train table?",
)
output_catalog = _get_param("output_catalog", "mlops_dev")
output_schema = _get_param("output_schema", "chenheju")
output_table = _get_param("output_table", "eda_agent_runs")

fq_table = f"{output_catalog}.{output_schema}.{output_table}"

# COMMAND ----------

mlflow.set_experiment(f"/Shared/experiments/eda_agent_{env}")

print(f"Env: {env}")
print(f"Git SHA: {git_sha}")
print(f"Run ID: {run_id}")
print(f"Question: {question!r}")
print(f"Output table: {fq_table}")

# COMMAND ----------

agent = EDAAgent()

session_id = f"job-{env}-{uuid.uuid4().hex[:8]}"
with mlflow.start_run(run_name=f"eda_agent_{env}_{run_id}") as run:
    mlflow.set_tags({
        "env": env,
        "git_sha": git_sha,
        "job_run_id": run_id,
        "session_id": session_id,
    })
    answer = agent.run(question, session_id=session_id)
    mlflow_run_id = run.info.run_id

print("\n" + "=" * 80)
print("AGENT ANSWER")
print("=" * 80)
print(answer)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

spark.sql(f"CREATE CATALOG IF NOT EXISTS {output_catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {output_catalog}.{output_schema}")

schema = StructType([
    StructField("ts", TimestampType(), False),
    StructField("env", StringType(), False),
    StructField("git_sha", StringType(), True),
    StructField("job_run_id", StringType(), True),
    StructField("session_id", StringType(), False),
    StructField("question", StringType(), False),
    StructField("answer", StringType(), False),
    StructField("mlflow_run_id", StringType(), True),
])

row = [(
    datetime.now(UTC),
    env,
    git_sha,
    run_id,
    session_id,
    question,
    answer,
    mlflow_run_id,
)]

df = spark.createDataFrame(row, schema)
(
    df.write.mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(fq_table)
)
print(f"\nAppended run to {fq_table}")
