"""Sentence-basiertes Chunking mit Overlap.

Portiert aus supermemoryai/supermemory v2/apps/cf-ai-backend/src/utils/chonker.ts
— dort mit `compromise` (JS), hier mit `pysbd` (Python, mehrsprachig,
inklusive robuster Handhabung deutscher Abkürzungen wie z.B., u.a., Dr.).

Strategie:
  1. Text in Sätze segmentieren (pysbd, sprachspezifisch).
  2. Sätze akkumulieren, bis max_chars erreicht → Chunk abschließen.
  3. Neuen Chunk mit Overlap (letzte overlap_ratio der Sätze) beginnen.
  4. Safeguard: Einzelner Satz > 2 * max_chars wird hart segmentiert.

Rückgabe: list[str] von Chunks. Für sehr kurze Texte (<= max_chars) wird
genau 1 Chunk = Originaltext zurückgegeben → kein Overhead für normale Notizen.
"""
from __future__ import annotations
from typing import Literal

import pysbd

# Cache: pysbd-Segmenter sind nicht threadsafe zwischen Sprachen, aber pro
# Sprache wiederverwendbar. Server ist single-threaded (asyncio), daher ok.
_SEGMENTERS: dict[str, pysbd.Segmenter] = {}

# Supported languages (pysbd 0.3+). Auto-fallback auf "en" bei unbekanntem Code.
SUPPORTED_LANGS = {"am", "ar", "bg", "da", "de", "el", "en", "es", "fa",
                   "fr", "hi", "hy", "it", "ja", "kk", "mr", "my", "nl",
                   "pl", "ru", "ur", "zh"}

Language = Literal["de", "en", "auto"]


def _get_segmenter(lang: str) -> pysbd.Segmenter:
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    if lang not in _SEGMENTERS:
        _SEGMENTERS[lang] = pysbd.Segmenter(language=lang, clean=False)
    return _SEGMENTERS[lang]


def detect_language(text: str) -> str:
    """Minimale Heuristik DE vs EN über Stopwort-Frequenz.

    Bewusst simpel und schnell (kein ML-Modell). Für Mischtexte oder andere
    Sprachen fällt die Auswahl auf "en" zurück — pysbd's englischer Segmenter
    liefert auf deutschem Text immer noch brauchbare Ergebnisse, nur die
    Abkürzungsliste ist kleiner.
    """
    low = f" {text.lower()} "
    de = sum(1 for w in ("der", "die", "das", "und", "ist", "nicht",
                         "mit", "für", "auf", "ein", "eine", "im", "zum")
             if f" {w} " in low)
    en = sum(1 for w in ("the", "is", "and", "of", "to", "in", "for",
                         "with", "on", "a", "an", "this", "that")
             if f" {w} " in low)
    return "de" if de > en else "en"


def chunk_text(
    text: str,
    max_chars: int = 800,
    overlap_ratio: float = 0.2,
    language: Language = "auto",
    min_chunks_threshold: int = 1000,
) -> list[str]:
    """Sentence-level chunking mit konfigurierbarem Overlap.

    Parameter:
        text: Der zu chunkende Volltext.
        max_chars: Ziel-Chunkgröße in Zeichen. Ein Chunk kann diese Grenze
            kurz überschreiten (bis zum nächsten Satzende).
        overlap_ratio: Anteil der Sätze, die in den nächsten Chunk übernommen
            werden (0.0 = kein Overlap, 0.5 = halber Chunk Overlap).
        language: "de", "en" oder "auto" (heuristische Erkennung).
        min_chunks_threshold: Wenn len(text) <= threshold → 1 Chunk zurück,
            kein Chunking-Overhead. Default 1000 = eine halbe A4-Seite.

    Raises:
        ValueError bei ungültigen Parametern.
    """
    if not text:
        return []
    if max_chars < 100:
        raise ValueError("max_chars must be >= 100")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0)")

    # Short-circuit: kurze Texte werden nicht gechunkt
    if len(text) <= min_chunks_threshold:
        return [text]

    lang = detect_language(text) if language == "auto" else language
    seg = _get_segmenter(lang)

    sentences = seg.segment(text)
    # pysbd gibt leere Strings oder whitespace-only zurück bei unüblichen
    # Eingaben — filtern
    sentences = [s for s in (s.strip() for s in sentences) if s]

    if not sentences:
        return [text]

    # Safeguard: extrem lange "Sätze" (z.B. Code, Base64, URLs ohne Spaces)
    # werden hart segmentiert, sonst explodiert max_chars.
    hard_limit = max_chars * 2
    flat: list[str] = []
    for s in sentences:
        if len(s) <= hard_limit:
            flat.append(s)
        else:
            # Schnitte an Whitespace wenn möglich
            for i in range(0, len(s), max_chars):
                flat.append(s[i:i + max_chars])

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for sent in flat:
        current.append(sent)
        current_size += len(sent) + 1  # +1 für Space-Joiner

        if current_size >= max_chars:
            chunks.append(" ".join(current))

            # Overlap vorbereiten
            overlap_n = max(1, int(len(current) * overlap_ratio))
            current = current[-overlap_n:] if overlap_ratio > 0 else []
            current_size = sum(len(s) + 1 for s in current)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_stats(text: str, **chunk_kwargs) -> dict:
    """Diagnose-Helper: Chunking-Ergebnis inspizieren.

    >>> stats = chunk_stats(long_text, max_chars=800)
    >>> stats["n_chunks"], stats["avg_size"], stats["oversize_chunks"]
    """
    chunks = chunk_text(text, **chunk_kwargs)
    if not chunks:
        return {"n_chunks": 0, "total_chars": 0}
    sizes = [len(c) for c in chunks]
    max_chars = chunk_kwargs.get("max_chars", 800)
    return {
        "n_chunks": len(chunks),
        "total_chars": sum(sizes),
        "avg_size": sum(sizes) // len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "oversize_chunks": sum(1 for s in sizes if s > max_chars * 1.5),
        "compression_ratio": round(sum(sizes) / len(text), 3),
    }
