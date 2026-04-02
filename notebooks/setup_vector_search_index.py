# Databricks notebook source
"""
Create or sync the Delta Sync Vector Search index on the metadata
knowledge table.

Uses databricks-sdk (pre-installed on Serverless).
"""

import time  # noqa: E402

from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.vectorsearch import (  # noqa: E402
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
    VectorIndexType,
)

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")  # noqa: F821
schema = dbutils.widgets.get("schema")  # noqa: F821
vs_endpoint = dbutils.widgets.get("vs_endpoint")  # noqa: F821
embedding_model = dbutils.widgets.get("embedding_model")  # noqa: F821

source_table = f"{catalog}.{schema}.metadata_knowledge"
index_name = f"{catalog}.{schema}.metadata_knowledge_index"

# COMMAND ----------

w = WorkspaceClient()

try:
    existing = w.vector_search_indexes.get_index(index_name)
    print(
        f"Index {index_name} already exists (ready={existing.status.ready}). Syncing..."
    )
    w.vector_search_indexes.sync_index(index_name)
except Exception:
    print(f"Creating Delta Sync index: {index_name}")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=vs_endpoint,
        primary_key="id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name=embedding_model,
                ),
            ],
        ),
    )
    print(f"Index {index_name} creation initiated.")

# COMMAND ----------

print(f"Waiting for index {index_name} to become ready...")
while True:
    idx = w.vector_search_indexes.get_index(index_name)
    if idx.status.ready:
        print("Index is READY.")
        break
    print("  status: not ready yet, waiting 30s...")
    time.sleep(30)

# COMMAND ----------

results = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["id", "section", "title", "content"],
    query_text="How do I join the tables together?",
    num_results=3,
)

print("Test query: 'How do I join the tables together?'\n")
for row in results.result.data_array:
    print(f"  [{row[0]}] {row[2]}")
    print(f"  {row[3][:120]}...\n")
