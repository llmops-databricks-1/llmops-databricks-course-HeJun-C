# Databricks notebook source
"""
Load raw CSV files from a UC Volume into managed Delta tables.
"""

import os
import sys

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()  # noqa: F821
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
src_path = os.path.normpath(f"/Workspace{os.path.dirname(notebook_path)}/../src")
sys.path.insert(0, src_path)

from llmops_databricks_course_HeJun_C.data_loading import (  # noqa: E402
    RawDataConfig,
    load_raw_tables,
)

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821

config = RawDataConfig(
    catalog=catalog,
    schema=schema,
    volume_path=f"/Volumes/{catalog}/{schema}/llmops_data",
)

# COMMAND ----------

load_raw_tables(spark, config)  # noqa: F821
