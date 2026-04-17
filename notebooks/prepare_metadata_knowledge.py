# Databricks notebook source
"""
Write metadata knowledge chunks to a managed Delta table and enable
Change Data Feed so the table can back a Delta Sync Vector Search index.
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

from llmops_databricks_course_HeJun_C.metadata_knowledge import (  # noqa: E402
    write_knowledge_table,
)

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821

full_table = write_knowledge_table(
    spark,  # noqa: F821
    catalog=catalog,
    schema=schema,
    table_name="metadata_knowledge",
)

# COMMAND ----------

display(spark.table(full_table))  # noqa: F821
