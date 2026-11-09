import base64
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

ROOT = Path(__file__).resolve().parents[1]
ASSET_LOCK = json.loads((ROOT / "assets.lock.json").read_text())
MODEL_IDS = Literal["e5", "minilm", "qwen3", "gemma2"]
ENCODINGS = Literal["float", "f32_base64"]
MIME_TYPES = Literal["image/png", "image/jpeg", "audio/wav", "video/mp4"]
MODALITIES = Literal["text", "image", "audio", "video", "control"]
MEDIA_ID = Annotated[str, Field(pattern=r"^media_[a-f0-9]{64}$")]
LAYOUT_ID = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
JOB_ID = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
INDEX = Annotated[StrictInt, Field(ge=0, le=2048)]
AXIS = Literal["all"] | Annotated[list[INDEX], Field(min_length=1, max_length=64)]

MAX_MEDIA_BYTES = 4 * 1024 * 1024
MAX_MEDIA_BASE64 = 4 * ((MAX_MEDIA_BYTES + 2) // 3)
MAX_TEXT_BYTES = 32 * 1024
MAX_INPUT_MEDIA_BYTES = 8 * 1024 * 1024
MAX_JOB_BYTES = 4 * 1024 * 1024
MAX_BATCH = 16
MAX_JOB = 1000
MAX_PAGE = 32

# Model dimensions and limits are for the precise artifacts in assets.lock.json.
MODELS = {
    "e5": {"name": "multilingual-e5-small", "dimension": 384, "max_tokens": 511,
           "layers": 12, "heads": 12, "pooling": "mean", "format_hint": "query: ... / passage: ..."},
    "minilm": {"name": "paraphrase-multilingual-MiniLM-L12-v2", "dimension": 384, "max_tokens": 128,
               "layers": 12, "heads": 12, "pooling": "mean", "format_hint": "Plain text"},
    "qwen3": {"name": "Qwen3-Embedding-0.6B", "dimension": 1024, "max_tokens": 1024,
              "layers": 28, "heads": 16, "pooling": "last", "format_hint": "Instruct: ...\nQuery: ... for instructed queries"},
    "gemma2": {"name": "EmbeddingGemma 2", "dimension": 768, "max_tokens": 2048,
               "layers": 24, "heads": 4, "pooling": "mean", "format_hint": "Use the model card's task prefix when needed"},
}

MEDIA_KINDS = {"image/png": "image", "image/jpeg": "image", "audio/wav": "audio", "video/mp4": "video"}


class Failure(Exception):
    def __init__(self, code, message, *, retryable=False):
        super().__init__(message)
        self.info = {"code": code, "message": message, "retryable": retryable}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextPart(StrictModel):
    type: Literal["text"]
    text: Annotated[str, Field(max_length=MAX_TEXT_BYTES)]


class MediaPart(StrictModel):
    type: Literal["image", "audio", "video"]
    media_id: MEDIA_ID


class Multipart(StrictModel):
    parts: Annotated[list[Annotated[TextPart | MediaPart, Field(discriminator="type")]],
                     Field(min_length=1, max_length=32)]


Input = Annotated[str, Field(max_length=MAX_TEXT_BYTES)] | Multipart


class AllTokens(StrictModel):
    kind: Literal["all"]


class TokenRange(StrictModel):
    kind: Literal["range"]
    start: INDEX
    end: INDEX


class ModalitySelector(StrictModel):
    kind: Literal["modality"]
    modality: MODALITIES


class PartSelector(StrictModel):
    kind: Literal["part"]
    part: Annotated[StrictInt, Field(ge=0, lt=32)]


Selector = Annotated[AllTokens | TokenRange | ModalitySelector | PartSelector, Field(discriminator="kind")]


class MaskRule(StrictModel):
    action: Literal["allow", "block"]
    layers: AXIS = "all"
    heads: AXIS = "all"
    query: Selector
    key: Selector


class AttentionMask(StrictModel):
    layout_id: LAYOUT_ID | None = None
    rules: Annotated[list[MaskRule], Field(max_length=64)]


class EmbeddingItem(StrictModel):
    input: Input
    mask: AttentionMask | None = None


BatchItem = EmbeddingItem | Input


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def normalized_item(item):
    if not isinstance(item, EmbeddingItem):
        item = EmbeddingItem(input=item)
    parts = [{"type": "text", "text": item.input}] if isinstance(item.input, str) else item.input.model_dump()["parts"]
    mask = item.mask.model_dump() if item.mask else None
    if mask and not mask["rules"] and mask["layout_id"] is None:
        mask = None
    if mask:
        for rule in mask["rules"]:
            for axis in ("layers", "heads"):
                if rule[axis] != "all":
                    rule[axis] = sorted(set(rule[axis]))
    # Check Unicode here so SQLite, native JSON and the request fingerprint agree.
    try:
        canonical(parts).encode()
    except UnicodeEncodeError as exc:
        raise Failure("invalid_input", "Input contains invalid Unicode") from exc
    return {"parts": parts, "mask": mask}


def render_embedding(result, encoding):
    if encoding == "f32_base64" or "embedding" not in result:
        return result
    vector = base64.b64decode(result["embedding"])
    return {**result, "encoding": "float", "embedding": list(struct.unpack(f"<{result['dimension']}f", vector))}
