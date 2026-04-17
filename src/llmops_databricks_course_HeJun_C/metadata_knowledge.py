"""Metadata knowledge layer for the churn-prediction dataset.

Each chunk is a self-contained piece of knowledge that will be embedded
and indexed via Mosaic AI Vector Search so a future agent can retrieve
relevant context at query time.
"""

from __future__ import annotations

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

KNOWLEDGE_SCHEMA = StructType(
    [
        StructField("id", StringType(), False),
        StructField("section", StringType(), False),
        StructField("title", StringType(), False),
        StructField("content", StringType(), False),
    ]
)

KNOWLEDGE_CHUNKS: list[dict[str, str]] = [
    {
        "id": "table_members",
        "section": "Tables Overview",
        "title": "members",
        "content": (
            "Table Name: members\n"
            "Description: User information. Not every user in the dataset "
            "is available in other datasets.\n"
            "Columns:\n"
            "- msno (string): user id\n"
            "- city (integer): city id, numbers represent cities\n"
            "- bd (integer): age. Has outlier values ranging from -7000 to "
            "2015; use judgement when filtering\n"
            "- gender (string): sex, values are 'male', 'female', or null. "
            "Null means 'unknown'\n"
            "- registered_via (integer): registration method\n"
            "- registration_init_time (date): initial registration date, "
            "format 2011-09-15\n"
            "Row Grain: each row is a unique user. msno is the primary key.\n"
            "Null Values: none except gender where null means 'unknown'."
        ),
    },
    {
        "id": "table_user_logs",
        "section": "Tables Overview",
        "title": "user_logs",
        "content": (
            "Table Name: user_logs\n"
            "Description: Daily user logs describing listening behaviors. "
            "Data collected until 2/28/2017.\n"
            "Columns:\n"
            "- msno (string): user id\n"
            "- date (date): date of streaming behavior, format 2011-09-15\n"
            "- num_25 (integer): songs played less than 25% of length\n"
            "- num_50 (integer): songs played 25%-50% of length\n"
            "- num_75 (integer): songs played 50%-75% of length\n"
            "- num_985 (integer): songs played 75%-98.5% of length\n"
            "- num_100 (integer): songs played over 98.5% of length\n"
            "- num_unq (integer): unique songs played\n"
            "- total_secs (double): total seconds played\n"
            "Row Grain: multiple rows per user, typically one row per user "
            "per day. No obvious primary key.\n"
            "Null Values: none."
        ),
    },
    {
        "id": "table_transactions",
        "section": "Tables Overview",
        "title": "transactions",
        "content": (
            "Table Name: transactions\n"
            "Description: Transactions of users up until 2/28/2017.\n"
            "Columns:\n"
            "- msno (string): user id\n"
            "- payment_method_id (integer): payment method\n"
            "- payment_plan_days (integer): membership plan length in days\n"
            "- plan_list_price (integer): list price of the plan\n"
            "- actual_amount_paid (integer): actual amount paid\n"
            "- is_auto_renew (integer): 1 = auto renew, 0 = not\n"
            "- transaction_date (date): transaction date, format 2011-09-15\n"
            "- membership_expire_date (date): membership expiry date, "
            "format 2011-09-15\n"
            "- is_cancel (integer): 1 = user canceled in this transaction, "
            "0 = did not cancel\n"
            "Row Grain: multiple rows per user, one row per transaction. "
            "No obvious primary key.\n"
            "Null Values: none."
        ),
    },
    {
        "id": "table_train",
        "section": "Tables Overview",
        "title": "train",
        "content": (
            "Table Name: train\n"
            "Description: Contains user ids and whether they churned for "
            "March 2017.\n"
            "Columns:\n"
            "- msno (string): user id\n"
            "- is_churn (integer): target variable. Churn is defined as the "
            "user not continuing the subscription within 30 days of "
            "expiration. 1 = churn, 0 = renewal.\n"
            "Row Grain: one row per user.\n"
            "Null Values: none."
        ),
    },
    {
        "id": "join_logic",
        "section": "Table Join Logic",
        "title": "How to join tables",
        "content": (
            "All tables join on the msno column.\n"
            "- train <-> members: roughly user-level join (one-to-one).\n"
            "- train <-> transactions: one-to-many (multiple transactions "
            "per user).\n"
            "- train <-> user_logs: one-to-many (multiple daily log rows "
            "per user).\n"
            "Avoid joining raw transactions and user_logs directly as it "
            "can multiply rows without useful results.\n"
            "Not all msno values exist in all tables; use inner join for "
            "analysis to only analyze users present in all relevant tables."
        ),
    },
    {
        "id": "business_rules",
        "section": "Business Rules",
        "title": "Business rules and data caveats",
        "content": (
            "- is_cancel is NOT the same as is_churn. is_cancel indicates "
            "a cancellation within a single transaction; is_churn is the "
            "target variable indicating the user did not renew within 30 "
            "days of membership expiration.\n"
            "- bd (age) has obvious bad/outlier values and should be "
            "treated carefully (values range from -7000 to 2015).\n"
            "- gender = null should be interpreted as 'unknown', not as "
            "missing data.\n"
            "- user_logs and transactions tables usually need aggregation "
            "to user level before joining for churn analysis."
        ),
    },
]


def write_knowledge_table(
    spark: SparkSession,
    catalog: str = "mlops_dev",
    schema: str = "chenheju",
    table_name: str = "metadata_knowledge",
) -> str:
    """Write knowledge chunks to a managed Delta table and enable CDF.

    Returns the fully qualified table name.
    """
    full_name = f"{catalog}.{schema}.{table_name}"

    logger.info("Creating knowledge DataFrame ({} chunks)", len(KNOWLEDGE_CHUNKS))
    df = spark.createDataFrame(KNOWLEDGE_CHUNKS, schema=KNOWLEDGE_SCHEMA)

    logger.info("Writing → {} (Delta, overwrite)", full_name)
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_name)
    )

    logger.info("Enabling Change Data Feed on {}", full_name)
    spark.sql(
        f"ALTER TABLE {full_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

    logger.info("Done: {} ({} rows)", full_name, len(KNOWLEDGE_CHUNKS))
    return full_name
