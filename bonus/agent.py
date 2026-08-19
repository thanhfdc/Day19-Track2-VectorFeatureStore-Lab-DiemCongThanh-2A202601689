from __future__ import annotations

import math
import re
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.embeddings import Embedder

COLLECTION = "bonus_memory"


@dataclass
class Memory:
    memory_id: int
    user_id: str
    text: str
    topic: str


class HybridMemoryAgent:
    """Minimal hybrid memory POC: vector episodic memory + profile features."""

    def __init__(self, top_k: int = 3) -> None:
        self.embedder = Embedder("fastembed")
        self.client = QdrantClient(":memory:")
        vector_config = models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
        self.client.create_collection(collection_name=COLLECTION, vectors_config=vector_config)
        self.top_k = top_k
        self._next_id = 0
        self._memories: list[Memory] = []
        self._profiles: dict[str, dict] = {"u_001": {
            "preferred_language": "vi/en mix", "reading_speed_wpm": 240,
            "topic_affinity": "cloud security", "active_hours": "21:00-23:30",
            "learning_goal": "Kubernetes and trustworthy RAG",
            "queries_last_hour": 0, "recent_topics": [],
        }}

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        chunks = self._chunk(text)
        if not chunks:
            return
        points = []
        for chunk, vector in zip(chunks, self.embedder.embed(chunks)):
            topic = self._guess_topic(chunk)
            memory = Memory(self._next_id, user_id, chunk, topic)
            self._memories.append(memory)
            payload = {"user_id": user_id, "text": chunk, "topic": topic}
            points.append(models.PointStruct(id=memory.memory_id, vector=vector.tolist(), payload=payload))
            self._next_id += 1
        self.client.upsert(collection_name=COLLECTION, points=points)

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        profile = self._profile(user_id)
        topic = self._guess_topic(query)
        profile["queries_last_hour"] += 1
        if topic not in profile["recent_topics"]:
            profile["recent_topics"].append(topic)
        profile["recent_topics"] = profile["recent_topics"][-5:]

        lines = [
            f"User: {user_id}",
            "Profile features:",
            f"- preferred_language: {profile['preferred_language']}",
            f"- reading_speed_wpm: {profile['reading_speed_wpm']}",
            f"- topic_affinity: {profile['topic_affinity']}",
            f"- active_hours: {profile['active_hours']}",
            f"- learning_goal: {profile['learning_goal']}",
            "Recent activity features:",
            f"- queries_last_hour: {profile['queries_last_hour']}",
            f"- recent_topics: {', '.join(profile['recent_topics'])}",
            "Top episodic memories:",
        ]
        hits = self._hybrid_search(query, user_id, self.top_k)
        if not hits:
            lines.append("- (no relevant memory found)")
        for rank, (memory, score) in enumerate(hits, 1):
            lines.append(f"{rank}. [{memory.topic}] score={score:.4f} {memory.text}")
        return "\n".join(lines)

    def _profile(self, user_id: str) -> dict:
        return self._profiles.setdefault(user_id, {
            "preferred_language": "vi", "reading_speed_wpm": 220,
            "topic_affinity": "general", "active_hours": "evening",
            "learning_goal": "learn efficiently", "queries_last_hour": 0,
            "recent_topics": [],
        })

    def _hybrid_search(self, query: str, user_id: str, top_k: int) -> list[tuple[Memory, float]]:
        depth = max(10, top_k * 3)
        scores: dict[int, float] = {}
        memory_by_id = {m.memory_id: m for m in self._memories if m.user_id == user_id}
        for hits in (self._vector_search(query, user_id, depth),
                     self._keyword_search(query, user_id, depth)):
            for rank, (memory, _raw_score) in enumerate(hits, 1):
                scores[memory.memory_id] = scores.get(memory.memory_id, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [(memory_by_id[mid], score) for mid, score in ranked]

    def _vector_search(self, query: str, user_id: str, depth: int) -> list[tuple[Memory, float]]:
        q_filter = models.Filter(must=[models.FieldCondition(
            key="user_id", match=models.MatchValue(value=user_id))])
        points = self.client.query_points(collection_name=COLLECTION,
                                          query=next(self.embedder.embed([query])).tolist(),
                                          query_filter=q_filter, limit=depth).points
        memories = {m.memory_id: m for m in self._memories}
        return [(memories[int(p.id)], float(p.score)) for p in points]

    def _keyword_search(self, query: str, user_id: str, depth: int) -> list[tuple[Memory, float]]:
        q_terms = set(self._tokens(query))
        rows = []
        for memory in self._memories:
            if memory.user_id != user_id:
                continue
            terms = self._tokens(memory.text)
            overlap = len(q_terms & set(terms))
            if overlap:
                rows.append((memory, overlap / math.sqrt(max(len(terms), 1))))
        return sorted(rows, key=lambda row: (-row[1], row[0].memory_id))[:depth]

    def _chunk(self, text: str, max_words: int = 70) -> list[str]:
        chunks = []
        for para in re.split(r"\n\s*\n", text.strip()):
            words = para.split()
            chunks.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
        return [chunk for chunk in chunks if chunk]

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[\wÀ-ỹ]+", text.lower())

    def _guess_topic(self, text: str) -> str:
        t = text.lower()
        for needle, topic in [
            ("kubernetes", "kubernetes"), ("autoscaling", "cloud"),
            ("tự động mở rộng", "cloud"), ("cloud", "cloud"),
            ("security", "security"), ("bảo mật", "security"),
            ("rag", "rag"), ("retrieval", "rag"),
        ]:
            if needle in t:
                return topic
        return "general"
