import argparse
import json
import os

from dotenv import load_dotenv

from llmops_databricks_course_HeJun_C.mcp_genie import GenieMcpClient, GenieMcpConfig


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        default=os.environ.get("GENIE_MCP_SERVER_URL", ""),
        required=not bool(os.environ.get("GENIE_MCP_SERVER_URL")),
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "llmops-course"),
    )
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument("--token", default=os.environ.get("DATABRICKS_TOKEN"))
    parser.add_argument(
        "--query",
        default="How many rows are in mlops_dev.chenheju.train?",
    )
    args = parser.parse_args()

    client = GenieMcpClient(
        GenieMcpConfig(
            server_url=args.server_url,
            profile=args.profile,
            host=args.host,
            token=args.token,
        )
    )

    tools = client.list_tools()
    print("Tools:")
    for t in tools:
        print("-", t["name"])

    result = client.ask(args.query)
    print("\nAsk result:")
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
