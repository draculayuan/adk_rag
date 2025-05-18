from vertexai import agent_engines
import vertexai
import argparse
import os

from src.agent.agent import rag_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create / update a Vertex AI agent-engine deployment."
    )

    parser.add_argument(
        "--staging-bucket",
        help=(
            "GCS URI used by Vertex AI as its staging bucket, "
            "e.g. gs://my-bucket[/optional/path]. "
            "If omitted, falls back to the STAGING_BUCKET env var."
        ),
        default=os.getenv("STAGING_BUCKET"),
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    vertexai.init(staging_bucket=args.staging_bucket)

    remote_app = agent_engines.create(
        agent_engine=rag_agent,
        requirements=[
            "google-cloud-aiplatform[agent_engines,adk,langchain,ag2,llama_index]==1.90.0",
            "google-cloud-firestore==2.20.2",
        ],
        extra_packages=["src/agent"],
    )

    print(f"Remote agent created: {remote_app.name}")


if __name__ == "__main__":
    main()
