from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def seed(agent: HybridMemoryAgent) -> None:
    agent.remember(
        """
        I read a Kubernetes note about pods, deployments, services, and horizontal
        pod autoscaling. The key idea was to scale replicas based on CPU or custom
        metrics when user traffic increases.

        A second note compared multi-region load balancing with single-region
        failover. Multi-region costs more but reduces downtime and latency for
        users far from the main region.
        """
    )
    agent.remember(
        """
        Cloud security checklist: enforce least privilege IAM, rotate secrets,
        audit public buckets, enable logging, and separate production from
        development accounts.

        For RAG systems, retrieval quality should be evaluated with exact,
        paraphrase, and mixed queries. Hybrid search with RRF is more stable
        than keyword-only or vector-only search.
        """
    )
    agent.remember(
        """
        Vietnamese users often mix English technical words with Vietnamese
        intent, for example "summary cloud security" or "Kubernetes tự động mở
        rộng". A multilingual embedding model helps with these paraphrases.
        """
    )


def main() -> None:
    agent = HybridMemoryAgent(top_k=3)
    seed(agent)
    queries = [
        "Tôi đã đọc gì về Kubernetes?",
        "Recommend đọc gì tiếp",
        "Tôi đang quan tâm gì gần đây?",
        "Tài liệu về tự động mở rộng hạ tầng?",
        "Cho tôi summary cloud security",
    ]
    for i, query in enumerate(queries, 1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        print(agent.recall(query))


if __name__ == "__main__":
    main()
