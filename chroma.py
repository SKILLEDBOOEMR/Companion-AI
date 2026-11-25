import chromadb
import uuid
import time
import re
from datetime import datetime, timezone
from time import perf_counter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChatbotMemory:
    def __init__(self, path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="cuda",
        )

        self.collection = self.client.get_or_create_collection(
            name="chatmemory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function,
        )

    def _format_with_time(self, text: str, ts: int) -> str:
        """Prepends readable UTC date so the embedding captures temporal context."""
        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"[{dt_str}] {text}"

    def add_memory(self, character_id: str, text: str, kind: str = None):
        now = int(time.time())
        metadata = {"character_id": character_id, "timestamp": now}
        if kind:
            metadata["kind"] = kind

        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[self._format_with_time(text, now)],
            metadatas=[metadata],
        )

    def add_memories(self, character_id: str, summary: str, dedupe_distance: float = 0.2) -> int:
        """Split a summary into one entry per labeled line. Returns how many were stored."""
        label = re.compile(
            r"^\[(user|assistant_answer|assistant_advice|agreed_plan)\]\s*(.+)$"
        )
        base_now = int(time.time())
        ids, docs, metas = [], [], []
        seen_texts = set()

        for idx, line in enumerate(summary.splitlines()):
            m = label.match(line.strip())
            if not m:
                continue
            kind, raw_text = m.group(1), m.group(2).strip()
            if not raw_text or raw_text in seen_texts:
                continue

            # Skip near-duplicates of what's already stored
            try:
                res = self.collection.query(
                    query_texts=[raw_text],
                    n_results=1,
                    where={"character_id": character_id},
                )
                dists = res["distances"][0]
                if dists and dists[0] < dedupe_distance:
                    continue
            except Exception:
                pass

            seen_texts.add(raw_text)
            entry_ts = base_now + idx  

            ids.append(str(uuid.uuid4()))
            docs.append(self._format_with_time(raw_text, entry_ts))
            metas.append({
                "character_id": character_id,
                "timestamp": entry_ts,
                "kind": kind,
            })

        if docs:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)
        return len(docs)

    def get_recent_memories(self, character_id: str, limit: int = 5, kind: str = None) -> list[dict]:
        """Fetch the N most recent memories tied to character_id (no semantic search)."""
        where_filter = {"character_id": character_id}
        if kind:
            where_filter = {
                "$and": [
                    {"character_id": character_id},
                    {"kind": kind},
                ]
            }

        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return []

        records = [
            {
                "id": doc_id,
                "text": doc,
                "timestamp": meta.get("timestamp", 0),
                "kind": meta.get("kind"),
            }
            for doc_id, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

        # Sort newest first and slice to limit
        records.sort(key=lambda x: x["timestamp"], reverse=True)
        return records[:limit]

    def recall_memory(
        self,
        character_id: str,
        query: str,
        limit: int = 5,
        max_distance: float = 0.6,
        recency_weight: float = 0.15,
    ) -> list[dict]:
        """Semantic search blended with a recency boost so newer facts win ties."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit * 3,  # Over-fetch so we can re-rank by recency
            where={"character_id": character_id},
        )

        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        now = time.time()
        relevant_memories = []

        for doc, metadata, distance in zip(documents, metadatas, distances):
            if distance >= max_distance:
                continue

            ts = metadata.get("timestamp", now)
            age_days = max(0.0, (now - ts) / 86400.0)
            # Gentle penalty as memories age (0.0 for brand new, up to recency_weight for old)
            age_penalty = recency_weight * (1.0 - (1.0 / (1.0 + age_days / 7.0)))
            blended_score = distance + age_penalty

            relevant_memories.append({
                "text": doc,
                "timestamp": ts,
                "kind": metadata.get("kind"),
                "distance": round(distance, 4),
                "blended_score": round(blended_score, 4),
            })

        relevant_memories.sort(key=lambda x: x["blended_score"])
        return relevant_memories[:limit]

mem = ChatbotMemory()

delete_ids = [
    "7781a6ff-8c2c-4aef-9694-96fc9eb9e547",
    "1d17fa8f-29eb-4358-ad35-eebcb29d1afd",
    "67e4b64d-c097-46bf-8099-e611fae49237",
    "12c3572a-f313-49e9-ad82-6dd8cdd780f4",
    "f435f6e5-5278-4eb3-9f51-b502af67377e",
    "5f36a138-56a8-45b5-9b99-c2c68bfb452b",
    "7426f4c7-6c52-4990-8072-fe46536bcf0a",
    "1e6a7a3d-a7ba-4bbf-8429-00271922975d",
    "88e26364-7d1a-4435-989d-7635959f7dc2",
    "0402920a-feb7-4829-9a44-36e8a67dd0b6",
    "2e4e3766-3bc1-4014-b1f8-e8a88418cb35",  # duplicate of f33ac284
]

before = mem.collection.count()
mem.collection.delete(ids=delete_ids)
print(f"{before} -> {mem.collection.count()} notes")