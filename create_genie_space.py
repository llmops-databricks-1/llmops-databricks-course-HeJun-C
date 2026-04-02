import json
import uuid
from databricks.sdk import WorkspaceClient

PROFILE = "llmops-course"
WAREHOUSE_ID = "9c9d5a71fb3773e4"

# Replace with your real table names
MEMBERS_TABLE = "mlops_dev.chenheju.members"
TRAIN_TABLE = "mlops_dev.chenheju.train"
TRANSACTIONS_TABLE = "mlops_dev.chenheju.transactions"
USER_LOGS_TABLE = "mlops_dev.chenheju.user_logs"
METADATA_KNOWLEDGE_TABLE = "mlops_dev.chenheju.metadata_knowledge"

w = WorkspaceClient(profile=PROFILE)


def _gen_id() -> str:
    # Genie requires lowercase 32-hex IDs (UUID without hyphens)
    return uuid.uuid4().hex


space_def = {
    "version": 2,
    "config": {
        "sample_questions": sorted(
            [
                {"id": _gen_id(), "question": ["What is the churn rate overall?"]},
                {
                    "id": _gen_id(),
                    "question": ["Show churn rate by registration method."],
                },
            ],
            key=lambda x: x["id"],
        )
    },
    "data_sources": {
        "tables": sorted(
            [
                {
                    "identifier": MEMBERS_TABLE,
                    "description": ["User profile table. One row per user."],
                },
                {
                    "identifier": TRAIN_TABLE,
                    "description": ["Churn label table. One row per user."],
                },
                {
                    "identifier": TRANSACTIONS_TABLE,
                    "description": ["Transaction history. Multiple rows per user."],
                },
                {
                    "identifier": USER_LOGS_TABLE,
                    "description": ["Listening logs. Multiple rows per user."],
                },
                {
                    "identifier": METADATA_KNOWLEDGE_TABLE,
                    "description": [
                        "Project knowledge base: table descriptions, join logic, and business rules."
                    ],
                },
            ],
            key=lambda x: x["identifier"],
        )
    }
}

space = w.genie.create_space(
    warehouse_id=WAREHOUSE_ID,
    title="KKBox Genie Smoke Test",
    description="Minimal Genie space to test connection only",
    serialized_space=json.dumps(space_def)
)

print("Genie space created successfully.")
print("Space ID:", space.space_id)