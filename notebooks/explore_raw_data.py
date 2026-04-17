# Databricks notebook source
"""
Quick exploration of the 4 raw Delta tables in mlops_dev.chenheju.
"""

# COMMAND ----------

CATALOG = "mlops_dev"
SCHEMA = "chenheju"
TABLES = ["members", "train", "transactions", "user_logs"]

# COMMAND ----------

for table in TABLES:
    full_name = f"{CATALOG}.{SCHEMA}.{table}"
    df = spark.table(full_name)  # noqa: F821
    print(f"\n{'=' * 60}")
    print(f"  {full_name}  —  {df.count():,} rows, {len(df.columns)} columns")
    print(f"{'=' * 60}")
    df.printSchema()
    display(df.limit(10))  # noqa: F821

# COMMAND ----------

for table in TABLES:
    full_name = f"{CATALOG}.{SCHEMA}.{table}"
    df = spark.table(full_name)  # noqa: F821
    print(f"\n--- {full_name} summary statistics ---")
    display(df.summary())  # noqa: F821

# COMMAND ----------

from pyspark.sql.functions import col  # noqa: E402
from pyspark.sql.functions import sum as spark_sum  # noqa: E402

for table in TABLES:
    full_name = f"{CATALOG}.{SCHEMA}.{table}"
    df = spark.table(full_name)  # noqa: F821
    print(f"\n--- {full_name} null counts ---")
    null_counts = df.select(
        [spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]
    )
    display(null_counts)  # noqa: F821
