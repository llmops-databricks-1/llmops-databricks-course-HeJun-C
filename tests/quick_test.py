"""Quick smoke tests for the Genie space and the course_LLM serving endpoint."""

import json
import os
import sys

from databricks.sdk import WorkspaceClient

os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", "llmops-course")

GENIE_SPACE_ID = "01f12d5f438713899fcb6ee851206636"
LLM_ENDPOINT = "course_LLM"


def test_genie(w: WorkspaceClient) -> bool:
    """Start a Genie conversation and send a simple question."""
    print("=" * 60)
    print("TEST 1: Genie Space")
    print("=" * 60)
    try:
        conversation = w.genie.start_conversation(
            space_id=GENIE_SPACE_ID,
            content="What tables are available?",
        )
        print(f"  Conversation ID : {conversation.conversation_id}")
        print(f"  Message ID      : {conversation.message_id}")
        print("  [PASS] Genie space is reachable.\n")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}\n")
        return False


def test_llm_endpoint(w: WorkspaceClient) -> bool:
    """Send a short chat-completion request to the serving endpoint."""
    print("=" * 60)
    print("TEST 2: LLM Serving Endpoint (course_LLM)")
    print("=" * 60)
    try:
        endpoint = w.serving_endpoints.get(LLM_ENDPOINT)
        print(f"  Endpoint name  : {endpoint.name}")
        print(f"  Endpoint state : {endpoint.state}")

        payload = {
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."},
            ],
            "max_tokens": 64,
        }
        resp_dict = w.api_client.do(
            "POST",
            f"/serving-endpoints/{LLM_ENDPOINT}/invocations",
            body=payload,
        )

        try:
            answer = resp_dict["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            answer = json.dumps(resp_dict, indent=2)[:300]

        print(f"  Model response  : {answer}")
        print("  [PASS] LLM endpoint is reachable.\n")
        return True
    except Exception as exc:
        print(f"  [FAIL] {exc}\n")
        return False


def main() -> None:
    w = WorkspaceClient()
    print(f"Workspace host: {w.config.host}\n")

    results = [
        test_genie(w),
        test_llm_endpoint(w),
    ]

    print("=" * 60)
    if all(results):
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
