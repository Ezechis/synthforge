"""Token counting with the bge-large-en-v1.5 tokenizer (corpus/query model).

Loads the cached ``tokenizer.json`` directly via the Rust ``tokenizers`` library
(NOT ``transformers``/``sentence_transformers`` -- those drag in a broken
sklearn DLL in this environment). The tokenizer.json carries the BERT WordPiece
vocab plus the post-processor that adds [CLS]/[SEP], so ``count_tokens``
reproduces the exact model input length (including the 2 special tokens).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

MODEL_NAME = "BAAI/bge-large-en-v1.5"
_CACHE_GLOB = (
    "models--BAAI--bge-large-en-v1.5/snapshots/*/tokenizer.json"
)


def _find_tokenizer_json() -> Path:
    """Locate the cached bge tokenizer.json (resolving the HF symlink)."""
    import os

    hub = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    # HF_HOME may or may not already include the 'hub' segment.
    candidates = list(hub.glob(_CACHE_GLOB)) + list((hub / "hub").glob(_CACHE_GLOB))
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Cached tokenizer.json for {MODEL_NAME} not found under {hub}. "
        "Expected an offline HF cache snapshot."
    )


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_find_tokenizer_json()))


def count_tokens(text: str) -> int:
    """Number of bge-large-en-v1.5 input ids for ``text`` (incl. special tokens)."""
    return len(_tokenizer().encode(text).ids)


if __name__ == "__main__":
    for s in ("Chain-of-thought prompting elicits reasoning.", "hello world"):
        print(f"{count_tokens(s):4d}  {s!r}")
