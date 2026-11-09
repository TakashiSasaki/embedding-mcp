import base64
import io
import json
import logging
import math
import os
import sqlite3
import subprocess
import time
import uuid
import wave
from fractions import Fraction
from hashlib import sha256

from PIL import Image, UnidentifiedImageError

from contracts import Failure, MAX_MEDIA_BYTES, MEDIA_KINDS, canonical, digest, render_embedding

MAX_UNFINISHED_JOBS = 8
MAX_RETAINED_JOBS = 128
MAX_STORED_MEDIA_BYTES = 512 * 1024 * 1024
ACTIVE = ("queued", "running", "canceling")
TERMINAL = ("completed", "canceled", "failed")
logger = logging.getLogger(__name__)


class Storage:
    def __init__(self, path, retention_seconds):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.ttl = retention_seconds
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, request_sha TEXT NOT NULL,
                model TEXT NOT NULL, identity TEXT NOT NULL, encoding TEXT NOT NULL, state TEXT NOT NULL,
                total INTEGER NOT NULL, done INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL, finished REAL, error TEXT
            );
            CREATE TABLE IF NOT EXISTS items (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                idx INTEGER NOT NULL, payload TEXT NOT NULL, result TEXT,
                PRIMARY KEY (job_id, idx)
            );
            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY, mime TEXT NOT NULL, sha TEXT NOT NULL,
                size INTEGER NOT NULL, data BLOB NOT NULL
            );
        """)
        with self.db:
            now = time.time()
            self.db.execute("UPDATE jobs SET state='queued', updated=? WHERE state='running'", (now,))
            self.db.execute("UPDATE jobs SET state='canceled', updated=?, finished=? WHERE state='canceling'", (now, now))
        self.prune()

    def close(self):
        self.db.close()

    def prune(self):
        with self.db:
            self.db.execute("DELETE FROM jobs WHERE finished <= ?", (time.time() - self.ttl,))

    def counts(self):
        return dict(self.db.execute("SELECT state, count(*) FROM jobs GROUP BY state"))

    def _lookup(self, job_id):
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or (row["finished"] is not None and row["finished"] + self.ttl <= time.time()):
            raise Failure("job_not_found", "No retained job with that ID")
        return row

    def status(self, job_id):
        row = self._lookup(job_id)
        result = {key: row[key] for key in ("model", "state", "total", "done", "created", "updated", "finished", "encoding")}
        result["job_id"] = row["id"]
        result["identity"] = json.loads(row["identity"])
        result["expires_at"] = None if row["finished"] is None else row["finished"] + self.ttl
        if row["error"]:
            result["error"] = json.loads(row["error"])
        return result

    def submit(self, key, model, identity, items, encoding):
        request_sha = digest([model, encoding, items])
        self.prune()
        with self.db:
            row = self.db.execute("SELECT id, request_sha FROM jobs WHERE request_key=?", (key,)).fetchone()
            if row:
                if row["request_sha"] != request_sha:
                    raise Failure("idempotency_conflict", "That idempotency key already names a different request")
                return {**self.status(row["id"]), "replayed": True}
            counts = self.counts()
            if sum(counts.get(state, 0) for state in ACTIVE) >= MAX_UNFINISHED_JOBS:
                raise Failure("queue_full", "The job queue is full", retryable=True)
            if sum(counts.values()) >= MAX_RETAINED_JOBS:
                raise Failure("storage_full", "The retained job limit has been reached", retryable=True)
            job_id, now = uuid.uuid4().hex, time.time()
            self.db.execute("""
                INSERT INTO jobs (id, request_key, request_sha, model, identity, encoding, state, total, created, updated)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
            """, (job_id, key, request_sha, model, canonical(identity), encoding, len(items), now, now))
            self.db.executemany("INSERT INTO items (job_id, idx, payload) VALUES (?, ?, ?)",
                                ((job_id, i, canonical(item)) for i, item in enumerate(items)))
        return {**self.status(job_id), "replayed": False}

    def pick(self):
        row = self.db.execute("""
            SELECT jobs.id, model, identity, idx, payload FROM jobs
            JOIN items ON items.job_id=jobs.id AND items.idx=jobs.done
            WHERE state IN ('queued', 'running')
            ORDER BY created, jobs.id LIMIT 1
        """).fetchone()
        if row is None:
            return None
        with self.db:
            self.db.execute("UPDATE jobs SET state='running', updated=? WHERE id=? AND state='queued'",
                            (time.time(), row["id"]))
        return row

    def finish_item(self, job_id, index, result):
        with self.db:
            row = self._lookup(job_id)
            if row["state"] not in ("running", "canceling") or index != row["done"]:
                raise RuntimeError("Unexpected job progress")
            count = index + 1
            state = "canceled" if row["state"] == "canceling" else "completed" if count == row["total"] else "running"
            now = time.time()
            self.db.execute("UPDATE items SET result=? WHERE job_id=? AND idx=?", (canonical(result), job_id, index))
            self.db.execute("UPDATE jobs SET done=?, state=?, updated=?, finished=? WHERE id=?",
                            (count, state, now, now if state in TERMINAL else None, job_id))

    def finish_job(self, job_id, state, error=None):
        with self.db:
            now = time.time()
            self.db.execute("UPDATE jobs SET state=?, error=?, updated=?, finished=? WHERE id=?",
                            (state, canonical(error) if error else None, now, now, job_id))

    def cancel(self, job_id, active_job):
        row = self._lookup(job_id)
        if row["state"] in ("queued", "running"):
            if active_job == job_id:
                with self.db:
                    self.db.execute("UPDATE jobs SET state='canceling', updated=? WHERE id=?", (time.time(), job_id))
            else:
                self.finish_job(job_id, "canceled")
        return self.status(job_id)

    def page(self, job_id, cursor, limit):
        status = self.status(job_id)
        prefix = f"{job_id}:"
        index = 0
        if cursor is not None:
            if not cursor.startswith(prefix) or not cursor[len(prefix):].isascii() or not cursor[len(prefix):].isdigit():
                raise Failure("invalid_cursor", "The cursor does not belong to this job")
            index = int(cursor[len(prefix):])
        if index > status["done"]:
            raise Failure("invalid_cursor", "The cursor is beyond the published items")
        rows = self.db.execute("""
            SELECT idx, result FROM items WHERE job_id=? AND idx>=? AND idx<?
            ORDER BY idx LIMIT ?
        """, (job_id, index, status["done"], limit))
        items = [{"index": row["idx"], **render_embedding(json.loads(row["result"]), status["encoding"])} for row in rows]
        end = index + len(items)
        more = end < status["done"] or status["state"] in ACTIVE
        return {**status, "items": items, "next_cursor": f"{prefix}{end}" if more else None}

    def register_media(self, mime, data):
        sha = sha256(data).hexdigest()
        media_id = f"media_{sha}"
        with self.db:
            row = self.db.execute("SELECT mime FROM media WHERE id=?", (media_id,)).fetchone()
            if row is not None and row["mime"] != mime:
                raise Failure("media_type_conflict", "This payload already has a different MIME type")
            if row is None:
                used = self.db.execute("SELECT coalesce(sum(size), 0) FROM media").fetchone()[0]
                if used + len(data) > MAX_STORED_MEDIA_BYTES:
                    raise Failure("storage_full", "The registered media limit has been reached")
                self.db.execute("INSERT INTO media VALUES (?, ?, ?, ?, ?)", (media_id, mime, sha, len(data), data))
        return {"media_id": media_id, "mime_type": mime, "sha256": sha, "bytes": len(data)}

    def get_media(self, media_id, *, data=False):
        columns = "mime, size, data" if data else "mime, size"
        row = self.db.execute(f"SELECT {columns} FROM media WHERE id=?", (media_id,)).fetchone()
        if not row:
            raise Failure("media_not_found", "Register the media before referencing its ID")
        return row


def validate_media(mime, encoded):
    try:
        data = base64.b64decode(encoded, validate=True)
        if not data or len(data) > MAX_MEDIA_BYTES:
            raise ValueError("The payload must contain at most 4 MiB")
        kind = MEDIA_KINDS[mime]
        if kind == "image":
            expected = "PNG" if mime == "image/png" else "JPEG"
            with Image.open(io.BytesIO(data), formats=[expected]) as image:
                if max(image.size) > 2048 or getattr(image, "is_animated", False):
                    raise ValueError("Use a still image with sides at most 2048 pixels")
                image.verify()
            with Image.open(io.BytesIO(data), formats=[expected]) as image:
                image.load()
        elif kind == "audio":
            with wave.open(io.BytesIO(data)) as audio:
                channels, width, rate, frames, *_ = audio.getparams()
                if channels not in (1, 2) or width != 2 or not 8000 <= rate <= 48000 or not 0 < frames <= 20 * rate:
                    raise ValueError("Use at most 20 seconds of 16-bit mono or stereo PCM WAV")
                if len(audio.readframes(frames)) != channels * width * frames:
                    raise ValueError("WAV data is truncated")
        else:
            validate_video(data)
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError, wave.Error,
            EOFError, ZeroDivisionError, subprocess.SubprocessError, KeyError, IndexError) as exc:
        logger.debug("Media validation rejected a %s payload", mime)
        raise Failure("invalid_media", "Media bytes do not match the declared format or its size limits") from exc
    return data


def validate_video(data):
    if data[4:8] != b"ftyp":
        raise ValueError("Expected an MP4 file")
    probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "cache,pipe",
                            "-read_ahead_limit", "-1", "-f", "mov", "-enable_drefs", "0", "-use_absolute_path", "0",
                            "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,duration,nb_frames:format=duration",
                            "-of", "json", "cache:pipe:0"], input=data, capture_output=True, timeout=6, check=True,
                           env={"PATH": os.defpath, "LANG": "C.UTF-8"})
    metadata = json.loads(probe.stdout)
    tracks = metadata["streams"]
    videos = [track for track in tracks if track["codec_type"] == "video"]
    if len(videos) != 1 or len(tracks) > 2:
        raise ValueError("Use one video track and at most one audio track")
    video = videos[0]
    if video["codec_name"] != "h264" or any(track["codec_type"] not in ("video", "audio") for track in tracks):
        raise ValueError("Use H.264 video in an MP4 container")
    duration = float(metadata.get("format", {}).get("duration", video.get("duration", "nan")))
    fps = Fraction(video["r_frame_rate"])
    frames = int(video.get("nb_frames", "0").replace("N/A", "0"))
    if (not math.isfinite(duration) or not 0 < duration <= 10 or not 0 < fps <= 60 or frames > 600
            or not 0 < video["width"] <= 1920 or not 0 < video["height"] <= 1080):
        raise ValueError("Use at most 10 seconds of MP4 with a maximum size of 1920 by 1080")
