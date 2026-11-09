import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import hmac
import json
import logging
import os
from pathlib import Path
import re
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn

from contracts import (
    AttentionMask, BatchItem, EmbeddingItem, Failure, Input, ROOT, canonical, normalized_item, render_embedding,
    ENCODINGS, JOB_ID, MAX_BATCH, MAX_JOB, MAX_JOB_BYTES, MAX_MEDIA_BASE64, MAX_PAGE, MIME_TYPES, MODEL_IDS, MODELS,
)
from runtime import Scheduler, Worker, asset_entry
from storage import Storage, validate_media

PUBLIC_HOST = "embedding-mcp.moukaeritai.work"
SERVICE_VERSION = "0.1.0"
logger = logging.getLogger(__name__)

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
CANCEL = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)


def fail_tool(info):
    raise ToolError(canonical(info))


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Failure as exc:
        fail_tool(exc.info)


def checked(result):
    if "error" in result:
        fail_tool(result["error"])
    return result


class StaticBearer:
    def __init__(self, app, token):
        self.app = app
        self.expected = b"Bearer " + token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            values = Headers(scope=scope).getlist("authorization")
            if len(values) != 1 or not hmac.compare_digest(values[0].encode("latin-1"), self.expected):
                response = JSONResponse({"error": "Bearer token required"}, 401, {"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def token_from_file(path):
    if path is None:
        return None
    with path.open("rb") as stream:
        if os.fstat(stream.fileno()).st_mode & 0o077:
            raise ValueError("The bearer token file must be private to the service user")
        token = stream.read(513).rstrip(b"\r\n")
    if not re.fullmatch(rb"[A-Za-z0-9._~+/-]{16,500}={0,2}", token):
        raise ValueError("The bearer token file must contain one token")
    return token


def version(worker_version):
    return f"{SERVICE_VERSION}+{worker_version.split('+', 1)[1]}"


def create_app(scheduler, port, token):
    worker, store = scheduler.worker, scheduler.storage
    security = TransportSecuritySettings(
        allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}", PUBLIC_HOST],
        allowed_origins=[f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"https://{PUBLIC_HOST}"],
    )
    mcp = FastMCP(
        "CPU Embedding Experiments",
        instructions=(
            "Inspect token positions before using raw ranges. Use model-specific text formatting where appropriate. "
            "Register media before passing its ID. Use durable jobs for hundreds of inputs."
        ),
        stateless_http=True, json_response=True, max_request_body_size=8 * 1024 * 1024,
        transport_security=security,
    )
    # FastMCP 1.30 forwards its low-level Server version in the MCP handshake.
    mcp._mcp_server.version = version(worker.version)
    media_validation = asyncio.Semaphore(1)

    @mcp.tool(annotations=READ)
    def list_models() -> dict[str, Any]:
        """List the fixed CPU models, token limits, dimensions, and supported media types."""
        models = []
        for model, spec in MODELS.items():
            media = model == "gemma2"
            models.append({
                "id": model, **spec, "model_sha256": asset_entry(model)["sha256"],
                "projector_sha256": asset_entry("gemma2-media")["sha256"] if media else None,
                "modalities": ["text", "image", "audio", "video"] if media else ["text"],
                "weights_present": (worker.assets / asset_entry(model)["path"]).is_file(),
                "projector_present": (worker.assets / asset_entry("gemma2-media")["path"]).is_file() if media else None,
                "attention_rules": {"actions": ["allow", "block"], "scopes": ["layer", "query_head"]},
            })
        return {"models": models}

    @mcp.tool(annotations=READ)
    def runtime_status() -> dict[str, Any]:
        """Read memory availability, the worker profile and PID, last unload reason, and queue state."""
        return scheduler.status()

    @mcp.tool(annotations=READ)
    async def inspect_input(model: MODEL_IDS, input: Input) -> dict[str, Any]:
        """Return the layout ID and half-open zero-based token spans for the expanded input."""
        item = call(normalized_item, input)
        return checked((await scheduler.interactive(model, [item], inspect=True))[0])

    @mcp.tool(annotations=READ)
    async def embed(model: MODEL_IDS, input: Input, mask: AttentionMask | None = None,
                    encoding: ENCODINGS = "float") -> dict[str, Any]:
        """Embed one text or mixed input using native pooling and optional ordered layer/head rules."""
        item = call(normalized_item, EmbeddingItem(input=input, mask=mask))
        result = checked((await scheduler.interactive(model, [item]))[0])
        return render_embedding(result, encoding)

    @mcp.tool(annotations=READ)
    async def embed_batch(model: MODEL_IDS, inputs: Annotated[list[BatchItem], Field(min_length=1, max_length=MAX_BATCH)],
                          encoding: ENCODINGS = "float") -> dict[str, Any]:
        """Embed up to 16 items in input order, returning a vector or an error per item."""
        results = await scheduler.interactive(model, [call(normalized_item, item) for item in inputs])
        return {"items": [{"index": i, **render_embedding(item, encoding)} for i, item in enumerate(results)]}

    @mcp.tool(annotations=WRITE)
    def submit_embedding_job(
        model: MODEL_IDS,
        inputs: Annotated[list[BatchItem], Field(min_length=1, max_length=MAX_JOB)],
        idempotency_key: Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")],
        encoding: ENCODINGS = "f32_base64",
    ) -> dict[str, Any]:
        """Persist up to 1,000 items for background embedding. Replay the same key and input to recover the job ID."""
        items = [call(normalized_item, item) for item in inputs]
        if len(canonical(items).encode()) > MAX_JOB_BYTES:
            fail_tool(Failure("invalid_input", "A job accepts up to 4 MiB of serialized inputs").info)
        media = model == "gemma2" and any(part["type"] != "text" for item in items for part in item["parts"])
        result = call(store.submit, idempotency_key, model, worker.identity(model, media), items, encoding)
        scheduler.wake.set()
        return result

    @mcp.tool(annotations=READ)
    def embedding_job_status(job_id: JOB_ID) -> dict[str, Any]:
        """Read persisted progress, job identity, and expiry."""
        return call(store.status, job_id)

    @mcp.tool(annotations=READ)
    def embedding_job_results(job_id: JOB_ID, cursor: Annotated[str, Field(max_length=45)] | None = None,
                              limit: Annotated[int, Field(ge=1, le=MAX_PAGE)] = 16) -> dict[str, Any]:
        """Read an ordered page of completed items. Replay the same cursor to recover a page."""
        return call(store.page, job_id, cursor, limit)

    @mcp.tool(annotations=CANCEL)
    def cancel_embedding_job(job_id: JOB_ID) -> dict[str, Any]:
        """Cancel a job at the next item boundary and retain all completed items."""
        return call(store.cancel, job_id, scheduler.active_job)

    @mcp.tool(annotations=WRITE)
    async def register_media(mime_type: MIME_TYPES, data_base64: Annotated[str, Field(max_length=MAX_MEDIA_BASE64)]) -> dict[str, Any]:
        """Validate and register up to 4 MiB of PNG, JPEG, WAV, or MP4 data. Returns a content-addressed media ID."""
        async with media_validation:
            try:
                data = await asyncio.to_thread(validate_media, mime_type, data_base64)
            except Failure as exc:
                fail_tool(exc.info)
            return call(store.register_media, mime_type, data)

    app = mcp.streamable_http_app()
    transport_security = TransportSecurityMiddleware(security)

    async def no_standalone_stream(request):
        rejection = await transport_security.validate_request(request)
        if rejection is not None:
            return rejection
        return Response(status_code=405, headers={"Allow": "POST"})

    # All tool responses use POST. Decline optional GET streams before FastMCP opens one.
    app.router.routes.insert(0, Route(mcp.settings.streamable_http_path, no_standalone_stream, methods=["GET", "HEAD"]))

    async def health(_):
        healthy = scheduler.task is not None and not scheduler.task.done()
        return JSONResponse({"status": "ok" if healthy else "unavailable",
                             "version": version(worker.version), "engine_version": worker.version},
                            status_code=200 if healthy else 503)

    app.add_route("/health", health, methods=["GET"])
    return StaticBearer(app, token) if token is not None else app


@contextmanager
def lock_state(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise ValueError("The state directory must be private to the service user")
    with (directory / "server.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another MCP server owns this state directory") from exc
        yield


async def serve(args):
    token = token_from_file(args.bearer_token_file)
    args.runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.runtime_dir.stat().st_mode & 0o077:
        raise ValueError("The worker socket directory must be private to the service user")
    with lock_state(args.state_dir):
        worker = Worker(args.worker, args.assets, args.runtime_dir, args.threads, args.idle_seconds, args.min_available_mib)
        storage = Storage(args.state_dir / "embeddings.sqlite3", args.job_ttl_seconds)
        try:
            async with Scheduler(worker, storage) as scheduler:
                app = create_app(scheduler, args.port, token)
                server = uvicorn.Server(uvicorn.Config(
                    app, host="127.0.0.1", port=args.port, access_log=False, proxy_headers=False,
                    limit_concurrency=64, timeout_graceful_shutdown=8,
                ))

                def scheduler_finished(task):
                    if not task.cancelled() and task.exception():
                        logger.error("Embedding scheduler exited", exc_info=task.exception())
                        server.should_exit = True

                scheduler.task.add_done_callback(scheduler_finished)
                logger.info("CPU Embedding MCP %s, engine %s", version(worker.version), worker.version)
                await server.serve()
        finally:
            storage.close()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description="Run the local CPU embedding MCP server")
    cache = Path(os.environ.get("EMBED_ASSET_CACHE", ROOT / ".assets"))
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", ROOT / ".assets"))
    parser.add_argument("--worker", type=Path, default=ROOT / ".assets/embedding-worker/current")
    parser.add_argument("--assets", type=Path, default=cache)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/embedding-mcp")
    parser.add_argument("--runtime-dir", type=Path, default=runtime_dir / "embedding-mcp")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--idle-seconds", type=float, default=300)
    parser.add_argument("--min-available-mib", type=int, default=3072)
    parser.add_argument("--job-ttl-seconds", type=int, default=86400)
    parser.add_argument("--bearer-token-file", type=Path, default=os.environ.get("EMBED_MCP_BEARER_FILE"))
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        engine = json.loads((args.worker / "manifest.json").read_text())["version"]
        print(f"{version(engine)} (llama.cpp {engine})")
        return
    if (not 1 <= args.port <= 65535 or not 1 <= args.threads <= 4 or not 1 <= args.idle_seconds <= 86400
            or args.min_available_mib < 0 or args.job_ttl_seconds < 1):
        parser.error("Invalid port, thread count, idle duration, memory threshold or job retention")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Each stateless request ends a transient transport and logs "Terminating session: None".
    logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
