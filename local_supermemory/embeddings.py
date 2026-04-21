"""Ollama Embedding Function für ChromaDB"""
import httpx
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import Optional


class OllamaEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-kompatible Embedding Function via Ollama API.
    
    Nutzt nomic-embed-text (768d) für hochwertige lokale Embeddings.
    Fallback auf ChromaDB-Default wenn Ollama nicht erreichbar.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0
    ):
        self.url = url
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Prüfe ob Ollama erreichbar ist und das Modell geladen."""
        if self._available is not None:
            return self._available
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Match auch ohne Tag (nomic-embed-text == nomic-embed-text:latest)
                self._available = any(
                    self.model in m or m.startswith(self.model)
                    for m in models
                )
                return self._available
        except Exception:
            pass
        self._available = False
        return False
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generiere Embeddings für eine Liste von Dokumenten."""
        if not self.is_available():
            raise RuntimeError(
                f"Ollama nicht verfügbar oder Modell '{self.model}' nicht geladen. "
                f"Bitte starte: ollama pull {self.model}"
            )
        
        embeddings: Embeddings = []
        for doc in input:
            try:
                r = httpx.post(
                    f"{self.url}/api/embeddings",
                    json={"model": self.model, "prompt": doc},
                    timeout=self.timeout
                )
                if r.status_code == 200:
                    emb = r.json().get("embedding")
                    if emb:
                        embeddings.append(emb)
                    else:
                        raise RuntimeError(f"Leere Embedding-Antwort für Dokument")
                else:
                    raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
            except httpx.ConnectError:
                self._available = False
                raise RuntimeError("Ollama-Verbindung verloren")
        
        return embeddings
    
    def reset_cache(self):
        """Zurücksetzen des Verfügbarkeits-Cache."""
        self._available = None


# Singleton
_ollama_ef: Optional[OllamaEmbeddingFunction] = None

def get_ollama_ef(model: str = "nomic-embed-text") -> OllamaEmbeddingFunction:
    global _ollama_ef
    if _ollama_ef is None or _ollama_ef.model != model:
        _ollama_ef = OllamaEmbeddingFunction(model=model)
    return _ollama_ef
