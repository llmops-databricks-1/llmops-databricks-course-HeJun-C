"""Raw data loading — reads CSVs from a UC Volume and writes managed Delta tables."""

from dataclasses import dataclass, field

from loguru import logger
from pyspark.sql import SparkSession

RAW_TABLES: list[str] = ["members", "train", "transactions", "user_logs"]


@dataclass
class RawDataConfig:
    catalog: str = "mlops_dev"
    schema: str = "chenheju"
    volume_path: str = "/Volumes/mlops_dev/chenheju/llmops_data"
    tables: list[str] = field(default_factory=lambda: list(RAW_TABLES))


def load_raw_tables(spark: SparkSession, config: RawDataConfig | None = None) -> None:
    """Read each CSV from a UC Volume and write it as a managed Delta table.

    Tables are written with mode=overwrite and overwriteSchema=True so the
    function is safe to re-run if the source CSV schema changes.
    """
    if config is None:
        config = RawDataConfig()

    for table in config.tables:
        csv_path = f"{config.volume_path}/{table}.csv"
        full_name = f"{config.catalog}.{config.schema}.{table}"

        logger.info("Reading '{}' from {}", table, csv_path)
        df = spark.read.csv(csv_path, header=True, inferSchema=True)

        logger.info(
            "Writing {} rows → {} (Delta, overwrite)",
            df.count(),
            full_name,
        )
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(full_name)
        )
        logger.info("Done: {}", full_name)
