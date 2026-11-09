import asyncio
import base64
from collections import deque
from dataclasses import dataclass
from hashlib import file_digest
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import tempfile
import time

import httpx

from contracts import (
    ASSET_LOCK, MEDIA_KINDS, MODELS, Failure, MAX_INPUT_MEDIA_BYTES, MAX_TEXT_BYTES, digest,
)

SYNC_DEADLINE = 80
JOB_ITEM_DEADLINE = 180
MAX_INTERACTIVE_QUEUE = 32
logger = logging.getLogger(__name__)


def available_mib():
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("Linux MemAvailable is unavailable")


def sha_file(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def asset_entry(model):
    return ASSET_LOCK["groups"][model][0]


def tool_version(program):
    output = subprocess.check_output([program, "-version"], text=True, timeout=5, env=clean_worker_env())
    return output.splitlines()[0]


def clean_worker_env():
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "OMP_DYNAMIC": "FALSE",
    }


def rss_mib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except FileNotFoundError:
        pass
    return None


class Worker:
    def __init__(self, release, assets, socket_root, threads, idle_seconds, min_available_mib):
        self.release = release.resolve()
        self.assets = assets.resolve()
        self.socket_root = socket_root.resolve()
        self.threads = threads
        self.idle_seconds = idle_seconds
        self.min_available_mib = min_available_mib
        self.env = clean_worker_env()
        self.manifest = json.loads((self.release / "manifest.json").read_text())
        self.version = self.manifest["version"]
        if (self.manifest["upstream_commit"] != ASSET_LOCK["runtime"]["commit"] or
                not re.fullmatch(r"0\.9\.0-dev\+local\.[0-9a-f]{12}(?:\.dirty)?", self.version) or
                digest(self.manifest["files"]) != self.manifest["artifact_id"]):
            raise RuntimeError("Unexpected llama.cpp worker manifest")
        self.engine_id = self.manifest["artifact_id"]
        self.media_tools = {tool: tool_version(tool) for tool in ("ffmpeg", "ffprobe")}
        self.verified = {}
        self.process = None
        self.directory = None
        self.client = None
        self.profile = None
        self.started_at = None
        self.last_used = None
        self.load_seconds = None
        self.last_unload = None
        self.phase = "unloaded"

    def identity(self, model, media):
        return {
            "engine_version": self.version,
            "engine_id": self.engine_id,
            "upstream_commit": self.manifest["upstream_commit"],
            "model": model,
            "model_sha256": asset_entry(model)["sha256"],
            "projector_sha256": asset_entry("gemma2-media")["sha256"] if media else None,
            "media_tools": self.media_tools if media else None,
            "threads": self.threads,
        }

    def _verify(self, path, sha, size=None):
        before = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_ctime_ns, before.st_mtime_ns, sha)
        if self.verified.get(path) == signature:
            return
        if (size is not None and before.st_size != size) or sha_file(path) != sha:
            raise ValueError(f"SHA-256 mismatch for {path.name}")
        after = path.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_ctime_ns, after.st_mtime_ns) != signature[:-1]:
            raise ValueError(f"Asset changed during verification: {path.name}")
        self.verified[path] = signature

    def _verify_profile(self, model, media):
        try:
            for name, sha in self.manifest["files"].items():
                self._verify(self.release / name, sha)
        except (ValueError, OSError) as exc:
            logger.error("Invalid worker release: %s", exc)
            raise Failure("worker_unavailable", "The configured llama.cpp build failed verification") from exc
        paths = []
        try:
            for item in ([model, "gemma2-media"] if media else [model]):
                entry = asset_entry(item)
                path = self.assets / entry["path"]
                self._verify(path, entry["sha256"], entry["bytes"])
                paths.append(str(path))
        except (ValueError, OSError) as exc:
            logger.error("Invalid model asset: %s", exc)
            raise Failure("model_unavailable", f"Verified weights for {model} are unavailable") from exc
        return paths

    async def ensure(self, model, media):
        if self.process and self.process.returncode is None and self.profile[0] == model and (self.profile[1] or not media):
            return
        if self.process:
            await self.stop("profile_change" if self.process.returncode is None else "worker_exit")

        self.phase = "loading"
        self.profile = (model, media)
        start = time.monotonic()
        paths = await asyncio.to_thread(self._verify_profile, model, media)
        self.directory = tempfile.TemporaryDirectory(prefix="worker-", dir=self.socket_root)
        socket = Path(self.directory.name) / "llama.sock"
        if len(os.fsencode(socket)) > 107:
            raise Failure("worker_unavailable", "Worker socket path is too long")
        spec = MODELS[model]
        tokens = str(spec["max_tokens"])
        command = [
            str(self.release / "llama-server"), "--host", str(socket), "-m", paths[0],
            "--embedding", "--embd-normalize", "2", "--pooling", spec["pooling"], "-ngl", "0",
            "-np", "1", "-fa", "off", "-t", str(self.threads), "-tb", str(self.threads),
            "-c", tokens, "-b", tokens, "-ub", tokens,
            "--no-cache-prompt", "--no-context-shift", "--no-webui", "--offline", "--log-verbosity", "2",
        ]
        if media:
            command += ["--mmproj", paths[1], "--no-mmproj-offload", "--video-fps", "2"]
        else:
            command += ["--no-mmproj"]
        env = {**self.env, "TMPDIR": self.directory.name}
        try:
            self.process = await asyncio.create_subprocess_exec(*command, cwd=self.directory.name, env=env,
                                                                 start_new_session=True)
        except OSError as exc:
            raise Failure("worker_unavailable", "Unable to start the configured llama.cpp worker", retryable=True) from exc
        self.client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://localhost",
                                        timeout=None, trust_env=False)
        async with asyncio.timeout(60):
            while self.process.returncode is None:
                try:
                    ready = await self.client.get("/health", timeout=1)
                    if ready.is_success:
                        self.load_seconds = round(time.monotonic() - start, 3)
                        self.started_at = time.time()
                        self.phase = "ready"
                        logger.info("Loaded worker %s for %s%s in %.3f seconds", self.version, model,
                                    " with media" if media else "", self.load_seconds)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
        raise Failure("worker_unavailable", f"llama.cpp exited while loading {model}", retryable=True)

    async def stop(self, reason):
        if self.phase == "unloaded":
            return
        self.phase = "unloading"
        if self.process:
            pgid = self.process.pid
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except TimeoutError:
                pass
            finally:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()
        if self.client:
            await self.client.aclose()
        if self.directory:
            self.directory.cleanup()
        self.process = self.client = self.directory = self.profile = self.started_at = self.last_used = None
        self.last_unload = {"reason": reason, "at": time.time()}
        self.phase = "unloaded"
        logger.info("Released worker memory (%s)", reason)

    async def maintain(self, memory):
        if self.process and self.process.returncode is not None:
            await self.stop("worker_exit")
        elif self.process and memory < self.min_available_mib:
            await self.stop("memory_pressure")
        elif self.last_used is not None and time.monotonic() - self.last_used >= self.idle_seconds:
            await self.stop("idle")

    async def request(self, model, media, body, inspect, deadline):
        try:
            async with asyncio.timeout_at(deadline):
                await self.ensure(model, media)
                response = await self.client.post("/local/layout" if inspect else "/local/embedding", json=body)
                self.last_used = time.monotonic()
            value = response.json()
        except TimeoutError as exc:
            await self.stop("deadline")
            raise Failure("deadline_exceeded", "The request exceeded its CPU deadline", retryable=True) from exc
        except (httpx.HTTPError, ValueError) as exc:
            await self.stop("worker_error")
            raise Failure("worker_unavailable", "The llama.cpp worker disconnected or returned invalid JSON",
                          retryable=True) from exc
        except Failure:
            await self.stop("load_error")
            raise
        if response.status_code == 400:
            error = value.get("error", value)
            raise Failure("invalid_input", error.get("message", "The model rejected the input or its mask"))
        if not response.is_success:
            raise Failure("compute_error", "The llama.cpp worker could not complete the embedding", retryable=True)
        return value

    def status(self):
        process = self.process
        return {
            "phase": self.phase,
            "model": self.profile[0] if self.profile else None,
            "media_profile": self.profile[1] if self.profile else False,
            "pid": process.pid if process and process.returncode is None else None,
            "rss_mib": rss_mib(process.pid) if process and process.returncode is None else None,
            "loaded_at": self.started_at,
            "load_seconds": self.load_seconds,
            "idle_seconds": None if self.last_used is None else round(time.monotonic() - self.last_used, 2),
            "idle_limit_seconds": self.idle_seconds,
            "min_available_mib": self.min_available_mib,
            "last_unload": self.last_unload,
            "engine_version": self.version,
            "engine_id": self.engine_id,
        }


@dataclass
class Interactive:
    model: str
    item: dict
    inspect: bool
    deadline: float
    future: asyncio.Future


class Scheduler:
    def __init__(self, worker, storage):
        self.worker = worker
        self.storage = storage
        self.queue = deque()
        self.wake = asyncio.Event()
        self.active_job = None
        self.active_request = None
        self.task = None

    async def __aenter__(self):
        self.task = asyncio.create_task(self.run())
        return self

    async def __aexit__(self, *_):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        finally:
            await self.worker.stop("shutdown")

    async def interactive(self, model, items, *, inspect=False):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SYNC_DEADLINE
        self.queue = deque(entry for entry in self.queue if not entry.future.done())
        if len(self.queue) + len(items) > MAX_INTERACTIVE_QUEUE:
            raise Failure("queue_full", "The interactive queue is full", retryable=True)
        futures = []
        for item in items:
            future = loop.create_future()
            futures.append(future)
            self.queue.append(Interactive(model, item, inspect, deadline, future))
        self.wake.set()
        try:
            await asyncio.wait(futures, timeout=SYNC_DEADLINE)
            missing = {"error": Failure("deadline_exceeded", "The request exceeded its CPU deadline", retryable=True).info}
            return [future.result() if future.done() else missing for future in futures]
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()

    def prepare(self, model, item, *, inspect=False):
        text_bytes = 0
        media_bytes = 0
        media_parts = 0
        parts = []
        for part in item["parts"]:
            if part["type"] == "text":
                text_bytes += len(part["text"].encode())
                parts.append(part)
                continue
            if model != "gemma2":
                raise Failure("unsupported_modality", f"{model} only accepts text input")
            media_parts += 1
            if media_parts > 4:
                raise Failure("invalid_input", "An item accepts up to four media parts")
            media = self.storage.get_media(part["media_id"], data=True)
            if MEDIA_KINDS[media["mime"]] != part["type"]:
                raise Failure("invalid_input", "The part type does not match the registered MIME type")
            media_bytes += media["size"]
            if media_bytes > MAX_INPUT_MEDIA_BYTES:
                raise Failure("invalid_input", "Total media per item must fit within 8 MiB")
            parts.append({**part, "data": base64.b64encode(media["data"]).decode("ascii")})

        if text_bytes > MAX_TEXT_BYTES or not (text_bytes or media_bytes):
            raise Failure("invalid_input", "Provide nonempty input with at most 32 KiB of UTF-8 text")
        spec = MODELS[model]
        identity = self.worker.identity(model, bool(media_bytes))
        # Thread count does not change tokenization. The artifact and media tools do.
        body = {"parts": parts, "max_tokens": spec["max_tokens"],
                "layout_namespace": digest({key: value for key, value in identity.items() if key != "threads"})}
        if item["mask"] and not inspect:
            mask = item["mask"]
            if mask["layout_id"] is not None:
                body["layout_id"] = mask["layout_id"]
            for rule in mask["rules"]:
                if mask["layout_id"] is None and any(rule[axis]["kind"] == "range" for axis in ("query", "key")):
                    raise Failure("invalid_input", "Inspect this input and include its layout_id to use raw ranges")
                for axis in ("layers", "heads"):
                    if rule[axis] != "all" and rule[axis][-1] >= spec[axis]:
                        raise Failure("invalid_input", f"Attention {axis} index is out of range for {model}")
            body["rules"] = mask["rules"]
        return body, identity

    async def evaluate(self, model, item, inspect, deadline):
        try:
            body, identity = self.prepare(model, item, inspect=inspect)
            value = await self.worker.request(model, identity["projector_sha256"] is not None, body, inspect, deadline)
            result = {**identity, "layout_id": value["layout_id"], "token_count": value["token_count"]}
            if inspect:
                return {**result, "spans": value["spans"], "layers": MODELS[model]["layers"], "heads": MODELS[model]["heads"]}
            vector = value["embedding"]
            dimension = MODELS[model]["dimension"]
            if (value["engine_version"] != self.worker.version or len(vector) != dimension or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector) or
                    not math.isclose(sum(v * v for v in vector), 1, abs_tol=0.001)):
                raise Failure("compute_error", "The worker returned an incompatible or invalid embedding")
            rules = body.get("rules")
            return {**result, "dimension": dimension, "encoding": "f32_base64",
                    "embedding": base64.b64encode(struct.pack(f"<{dimension}f", *vector)).decode("ascii"),
                    "mask_id": digest([result["layout_id"], rules]) if rules else None}
        except Failure as exc:
            return {"error": exc.info}
        except (KeyError, TypeError, ValueError, OSError):
            logger.exception("Invalid worker result")
            await self.worker.stop("worker_error")
            return {"error": Failure("compute_error", "Unable to validate the worker result", retryable=True).info}

    async def run_job_item(self, row):
        model, job_id, index = row["model"], row["id"], row["idx"]
        self.active_job = job_id
        try:
            identity = json.loads(row["identity"])
            if model not in MODELS or identity != self.worker.identity(model, identity["projector_sha256"] is not None):
                self.storage.finish_job(job_id, "failed", Failure(
                    "identity_changed", "The saved engine or model identity differs from this deployment").info)
                return
            item = json.loads(row["payload"])
            result = await self.evaluate(model, item, False, asyncio.get_running_loop().time() + JOB_ITEM_DEADLINE)
            error = result.get("error")
            if error and error["code"] in ("model_unavailable", "worker_unavailable"):
                state = "canceled" if self.storage.status(job_id)["state"] == "canceling" else "failed"
                self.storage.finish_job(job_id, state, error if state == "failed" else None)
            else:
                self.storage.finish_item(job_id, index, result)
        finally:
            self.active_job = None

    async def run(self):
        next_prune = 0.0
        while True:
            now = time.monotonic()
            if now >= next_prune:
                self.storage.prune()
                next_prune = now + 30
            self.wake.clear()
            memory = available_mib()
            await self.worker.maintain(memory)
            if self.queue:
                entry = self.queue.popleft()
                if entry.future.done():
                    continue
                self.active_request = entry.model
                try:
                    if entry.deadline <= asyncio.get_running_loop().time():
                        result = {"error": Failure("deadline_exceeded", "The request expired in the queue", retryable=True).info}
                    elif available_mib() < self.worker.min_available_mib:
                        result = {"error": Failure("memory_pressure", "Host memory is below the worker admission limit",
                                                   retryable=True).info}
                    else:
                        result = await self.evaluate(entry.model, entry.item, entry.inspect, entry.deadline)
                    if not entry.future.done():
                        entry.future.set_result(result)
                finally:
                    self.active_request = None
                continue
            if memory >= self.worker.min_available_mib:
                row = self.storage.pick()
                if row is not None:
                    await self.run_job_item(row)
                    continue
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1)
            except TimeoutError:
                pass

    def status(self):
        memory = available_mib()
        return {**self.worker.status(), "mem_available_mib": memory, "memory_pressure": memory < self.worker.min_available_mib,
                "interactive_queued": sum(not entry.future.done() for entry in self.queue),
                "active_request_model": self.active_request, "active_job": self.active_job,
                "job_counts": self.storage.counts(), "scheduler_healthy": self.task is not None and not self.task.done()}
