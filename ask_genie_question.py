from databricks.sdk import WorkspaceClient

PROFILE = "llmops-course"
SPACE_ID = "01f12c9cf75e1124a4f129b044f6a2a1"

w = WorkspaceClient(profile=PROFILE)

response = w.genie.start_conversation_and_wait(
    space_id=SPACE_ID,
    content="What ratio of male vs. female users are there in the members table?",
)

print(response)
