import os
from typing import List, Dict, Optional


class VectorStore:
    """Optional Chroma-backed vector store. No-op if deps missing."""

    def __init__(self, persist_dir: str = "vector_store", collection_name: str = "news"):
        self.enabled = False
        self.client = None
        self.collection = None
        self.embedder = None

        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(collection_name)
            # Multilingual, small and fast
            self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.enabled = True
        except Exception as exc:  # Missing deps or runtime error => stay disabled
            print(f"[VectorStore] Disabled: {exc}")
            self.enabled = False

    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> None:
        if not self.enabled or not documents:
            return
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self._encode(documents)
        self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        if not self.enabled or not query_text:
            return []
        emb = self._encode([query_text])[0]
        res = self.collection.query(query_embeddings=[emb], n_results=n_results)
        results = []
        for i in range(len(res.get("documents", [[]])[0])):
            results.append({
                "document": res["documents"][0][i],
                "metadata": res.get("metadatas", [[{}]])[0][i],
                "id": res.get("ids", [[None]])[0][i],
                "distance": res.get("distances", [[None]])[0][i],
            })
        return results

    def _encode(self, texts: List[str]):
        if not self.enabled:
            return []
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def list_all(self, offset: int = 0, limit: int = 50) -> Dict:
        """List documents with pagination. Returns {total, items}.
        Notes: Uses where={} and where_document={} to fetch all; may be batched by Chroma internally.
        """
        if not self.enabled:
            return {"total": 0, "items": []}
        try:
            # Chroma get supports pagination via limit/offset
            got = self.collection.get(limit=limit, offset=offset, include=["documents", "metadatas", "ids"])
            total = got.get("total", None)
            # Some versions may not return total; estimate crudely
            if total is None:
                total = offset + len(got.get("ids", []))
            items = []
            docs = got.get("documents", []) or []
            metas = got.get("metadatas", []) or []
            ids = got.get("ids", []) or []
            for i in range(len(ids)):
                items.append({
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                })
            return {"total": total, "items": items}
        except Exception as exc:
            print(f"[VectorStore] list_all failed: {exc}")
            return {"total": 0, "items": []}


# Singleton accessor
_singleton: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _singleton
    if _singleton is None:
        _singleton = VectorStore()
    return _singleton


