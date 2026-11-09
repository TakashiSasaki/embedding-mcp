"""Exercise the native engine and MCP service with the pinned CPU artifacts."""
import argparse
import asyncio
import base64
from contextlib import asynccontextmanager, contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from contracts import MODELS, ROOT, digest, normalized_item
from runtime import Worker, Scheduler, asset_entry, clean_worker_env
from storage import Storage
import runtime

sys.path.insert(0, str(ROOT / "experiments"))
from multimodal import audio, image, video  # reuse the tracked synthetic fixtures


def media_examples():
    return [
        ("image", "image/png", image((255, 0, 0))),
        ("audio", "audio/wav", audio(440)),
        ("video", "video/mp4", video("blue")),
    ]


def b64(data):
    return base64.b64encode(data).decode("ascii")


def span(position):
    return {"kind": "range", "start": position, "end": position + 1}


def rule(action, layers, heads, query, key):
    return {"action": action, "layers": layers, "heads": heads, "query": query, "key": key}


def probe(layer, head, query, key):
    return {"layer": layer, "head": head, "query": query, "key": key}


def max_delta(left, right):
    assert len(left) == len(right)
    return max(abs(a - b) for a, b in zip(left, right))


def assert_parity(left, right):
    delta = max_delta(left, right)
    assert delta <= 3e-6, delta
    return delta


@contextmanager
def engine(binary, assets, model, media=False):
    spec = MODELS[model]
    with tempfile.TemporaryDirectory(prefix="emb-qa-") as directory:
        path = Path(directory)
        uds = str(path / "llama.sock")
        limit = str(spec["max_tokens"])
        args = [str(binary), "--host", uds, "-m", str(assets / asset_entry(model)["path"]), "--embedding",
                "--pooling", spec["pooling"], "--embd-normalize", "2", "-np", "1", "-ngl", "0", "-fa", "off",
                "-t", "4", "-tb", "4", "-c", limit, "-b", limit, "-ub", limit, "--offline",
                "--no-cache-prompt", "--no-context-shift", "--no-webui", "--log-verbosity", "2"]
        if media:
            args += ["--mmproj", str(assets / asset_entry("gemma2-media")["path"]), "--no-mmproj-offload", "--video-fps", "2"]
        else:
            args += ["--no-mmproj"]
        with (path / "worker.log").open("w") as log:
            process = subprocess.Popen(args, stdout=log, stderr=log, env=clean_worker_env(), start_new_session=True)
            try:
                with httpx.Client(transport=httpx.HTTPTransport(uds=uds), base_url="http://localhost", timeout=120) as client:
                    wait_http(client, process, "/health")
                    yield client
            except BaseException:
                print((path / "worker.log").read_text()[-10000:], file=sys.stderr)
                raise
            finally:
                stop(process)


def stop(process, *, crash=False, worker_pid=None):
    if process.poll() is None:
        if crash:
            if worker_pid:
                try:
                    os.killpg(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.kill()
        else:
            process.terminate()
    try:
        process.wait(15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(5)


def wait_http(client, process, endpoint):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and process.poll() is None:
        try:
            if client.get(endpoint, timeout=1).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Local service did not become ready, exit={process.poll()}")


def post(client, endpoint, body):
    response = client.post(endpoint, json=body)
    assert response.is_success, response.text
    return response.json()


def native_checks(assets, release):
    stock = assets / "llama-b12110/llama-server"
    patched = release / "llama-server"
    evidence = {}
    text = "query: 日本語の埋め込みと English embeddings"
    text_part = {"type": "text", "text": text}
    for model, spec in MODELS.items():
        body = {"parts": [text_part], "layout_namespace": "qa-text", "max_tokens": spec["max_tokens"]}
        with engine(stock, assets, model) as client:
            stock_vector = post(client, "/v1/embeddings", {"input": text})["data"][0]["embedding"]
        with engine(patched, assets, model) as client:
            layout = post(client, "/local/layout", body)
            key = layout["token_count"] - 1
            body = {**body, "layout_id": layout["layout_id"],
                    "probes": [probe(0, 0, 0, key), probe(0, 1, 0, key), probe(1, 0, 0, key)]}
            base = post(client, "/local/embedding", body)
            action = "allow" if model == "qwen3" else "block"
            edit = rule(action, [0], [0], span(0), span(key))
            masked = post(client, "/local/embedding", {**body, "rules": [edit]})
            target, other_head, other_layer = masked["probes"]
            assert target["allowed"] == (action == "allow")
            assert (target["probability"] > 0) == (action == "allow")
            assert target["allowed"] != base["probes"][0]["allowed"]
            assert other_head == base["probes"][1]
            assert other_layer["allowed"] == base["probes"][2]["allowed"]
            assert max_delta(base["embedding"], masked["embedding"]) > 0
            invalid = client.post("/local/embedding", json={**body, "rules": [
                rule("block", [0], [0], span(0), {"kind": "all"}),
            ]})
            assert invalid.status_code == 400 and "no visible key" in invalid.text
            restore = post(client, "/local/embedding", body)
            assert restore["embedding"] == base["embedding"]
            # Opposite rules targeting exactly the same edge are ordered.
            reverse = {**edit, "action": "block" if action == "allow" else "allow"}
            last_wins = post(client, "/local/embedding", {**body, "rules": [reverse, edit]})
            assert last_wins["embedding"] == masked["embedding"]
            evidence[model] = {
                "dimension": len(stock_vector), "tokens": layout["token_count"],
                "stock_max_abs_error": assert_parity(stock_vector, base["embedding"]),
                "native_target": base["probes"][0], "edited_target": target,
                "edited_vector_max_delta": max_delta(base["embedding"], masked["embedding"]),
            }
            assert len(stock_vector) == spec["dimension"]
        print(f"Native text parity and attention passed for {model}", flush=True)

    parts, legacy = [text_part], [text_part]
    for i, (kind, mime, data) in enumerate(media_examples(), 1):
        encoded = b64(data)
        parts.append({"type": kind, "media_id": f"fixture-{i}", "data": encoded})
        if kind == "image":
            legacy.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        else:
            legacy.append({"type": f"input_{kind}", f"input_{kind}": {"data": encoded}})
    tail = {"type": "text", "text": " media examples / メディアの例"}
    parts.append(tail)
    legacy.append(tail)
    with engine(stock, assets, "gemma2", media=True) as client:
        stock_vector = post(client, "/v1/embeddings", {"input": [{"content": legacy}]})["data"][0]["embedding"]
    with engine(patched, assets, "gemma2", media=True) as client:
        body = {"parts": parts, "layout_namespace": "qa-media", "max_tokens": 2048}
        layout = post(client, "/local/layout", body)
        spans = layout["spans"]
        assert [part["start"] for part in spans] == [0, *[part["end"] for part in spans[:-1]]]
        assert spans[-1]["end"] == layout["token_count"]
        query = next(part["start"] for part in spans if part["parts"] == [4] and part["modality"] == "text")
        targets = {kind: next(part["start"] for part in spans if part["modality"] == kind
                              and part["parts"] == [i]) for i, kind in enumerate(("image", "audio", "video"), 1)}
        assert len([part for part in spans if part["modality"] == "video" and part["parts"] == [3]]) >= 2
        body["probes"] = [probe(layer, head, query, key) for key in targets.values()
                          for layer, head in ((0, 0), (0, 1), (5, 0))]
        base = post(client, "/local/embedding", body)
        rules = [rule("block", [0, 5], [0], {"kind": "part", "part": 4},
                      {"kind": "modality", "modality": kind}) for kind in targets]
        masked = post(client, "/local/embedding", {**body, "rules": rules})
        for baseline, edited in zip(base["probes"], masked["probes"]):
            assert baseline["allowed"] and baseline["probability"] > 0
            if edited["head"] == 0:
                assert not edited["allowed"] and edited["probability"] == 0
            else:
                assert baseline == edited
        # All heads and layers share each distinct native attention plane.
        all_layers = post(client, "/local/embedding", {**body, "rules": [
            rule("block", "all", "all", {"kind": "part", "part": 4}, {"kind": "modality", "modality": "image"}),
        ]})
        assert all(not p["allowed"] and p["probability"] == 0 for p in all_layers["probes"][:3])
        evidence["gemma2_mixed"] = {
            "tokens": layout["token_count"], "modality_targets": targets, "text_query": query,
            "stock_max_abs_error": assert_parity(stock_vector, base["embedding"]),
            "edited_vector_max_delta": max_delta(base["embedding"], masked["embedding"]),
            "native": base["probes"], "edited": masked["probes"], "all_layers": all_layers["probes"][:3],
        }
    print("Gemma 2 image/audio/video parity and local/global attention passed", flush=True)
    return evidence


@asynccontextmanager
async def mcp_client(url):
    async with httpx.AsyncClient(timeout=100, trust_env=False) as client:
        async with streamable_http_client(url, http_client=client, terminate_on_close=False) as (read, write, _):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                yield session, initialized.serverInfo.version


async def call(session, name, *, error=None, **args):
    response = await session.call_tool(name, args)
    if error:
        assert response.isError and error in str(response.content), response
        return
    assert not response.isError and response.structuredContent is not None, response
    return response.structuredContent


async def until(get, predicate, timeout=45):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = await get()
        if predicate(value):
            return value
        await asyncio.sleep(0.2)
    raise TimeoutError("Condition not met during verification")


def choose_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def service_checks(assets, release, items):
    with tempfile.TemporaryDirectory(prefix="embedding-mcp-qa-") as tmp:
        directory = Path(tmp)
        port = choose_port()
        url = f"http://127.0.0.1:{port}"
        log_path = directory / "service.log"

        def start(ttl=86400):
            with log_path.open("a") as log:
                process = subprocess.Popen([
                    sys.executable, str(ROOT / "service/main.py"), "--worker", str(release), "--assets", str(assets),
                    "--state-dir", str(directory / "state"), "--runtime-dir", str(directory / "runtime"),
                    "--port", str(port), "--idle-seconds", "2", "--job-ttl-seconds", str(ttl),
                ], stdout=log, stderr=log, start_new_session=True)
            try:
                with httpx.Client(base_url=url, trust_env=False) as client:
                    wait_http(client, process, "/health")
            except BaseException:
                stop(process)
                raise
            return process

        process = start()
        try:
            async with mcp_client(url + "/mcp") as (session, version):
                assert len((await session.list_tools()).tools) == 10
                assert (await call(session, "runtime_status"))["pid"] is None
                assert len((await call(session, "list_models"))["models"]) == 4
                assert (await call(session, "runtime_status"))["pid"] is None
                original = "日本語の埋め込み"
                layout = await call(session, "inspect_input", model="minilm", input=original)
                warm = await call(session, "runtime_status")
                mask = {"layout_id": layout["layout_id"], "rules": [rule("block", [0], [0], span(0), span(1))]}
                await call(session, "embed", model="minilm", input="different", mask=mask, error="layout_id")
                batch = await call(session, "embed_batch", model="minilm", inputs=[
                    original, "", {"input": original, "mask": mask},
                ])
                assert [item["index"] for item in batch["items"]] == [0, 1, 2]
                base, bad, edited = batch["items"]
                assert base["mask_id"] is None and bad["error"]["code"] == "invalid_input" and edited["mask_id"]
                assert max_delta(base["embedding"], edited["embedding"]) > 0
                assert (await call(session, "runtime_status"))["pid"] == warm["pid"]
                idle = await until(lambda: call(session, "runtime_status"), lambda r: r["pid"] is None, timeout=8)
                assert idle["last_unload"]["reason"] == "idle"
                assert (await call(session, "embed", model="minilm", input=original))["embedding"] == base["embedding"]
                reloaded = await call(session, "runtime_status")
                assert reloaded["pid"] != warm["pid"]

                gemma_text = await call(session, "embed", model="gemma2", input=original)
                gemma_before = await call(session, "runtime_status")
                assert not gemma_before["media_profile"]
                mixed = {"parts": [{"type": "text", "text": original}]}
                for kind, mime, data in media_examples():
                    registered = await call(session, "register_media", mime_type=mime, data_base64=b64(data))
                    mixed["parts"].append({"type": kind, "media_id": registered["media_id"]})
                mixed_layout = await call(session, "inspect_input", model="gemma2", input=mixed)
                assert {"text", "image", "audio", "video"}.issubset({part["modality"] for part in mixed_layout["spans"]})
                gemma_after = await call(session, "runtime_status")
                assert gemma_after["media_profile"] and gemma_after["pid"] != gemma_before["pid"]
                assert (await call(session, "embed", model="gemma2", input=original))["embedding"] == gemma_text["embedding"]
                assert (await call(session, "runtime_status"))["pid"] == gemma_after["pid"]

                args = {"model": "minilm", "inputs": [f"埋め込み {i % 17}" for i in range(items)], "idempotency_key": "qa-restart"}
                receipt = await call(session, "submit_embedding_job", **args)
                job_id = receipt["job_id"]
                await call(session, "submit_embedding_job", **{**args, "inputs": ["changed"]}, error="idempotency_conflict")
                status = await until(lambda: call(session, "embedding_job_status", job_id=job_id),
                                     lambda r: 3 <= r["done"] < items)
                before = (await call(session, "embedding_job_results", job_id=job_id, limit=3))["items"]
                worker_pid = (await call(session, "runtime_status"))["pid"]
            stop(process, crash=True, worker_pid=worker_pid)

            process = start()
            async with mcp_client(url + "/mcp") as (session, _):
                same = await call(session, "submit_embedding_job", **args)
                assert same["job_id"] == job_id and same["replayed"] and same["done"] >= status["done"]
                restored = (await call(session, "embedding_job_results", job_id=job_id, limit=3))["items"]
                assert restored == before
                final = await until(lambda: call(session, "embedding_job_status", job_id=job_id),
                                    lambda r: r["state"] in ("completed", "failed"), timeout=600)
                assert final["state"] == "completed" and final["done"] == items and final["expires_at"] - final["finished"] == 86400
                cursor, results = None, []
                while True:
                    page = await call(session, "embedding_job_results", job_id=job_id, cursor=cursor, limit=32)
                    results += page["items"]
                    cursor = page["next_cursor"]
                    if cursor is None:
                        break
                assert [item["index"] for item in results] == list(range(items))
                assert all(item["dimension"] == 384 and len(base64.b64decode(item["embedding"])) == 1536 for item in results)
                assert results[0]["embedding"] == results[17]["embedding"]
                assert results[0]["embedding"] != results[1]["embedding"]
                cancel_args = {**args, "idempotency_key": "qa-cancel"}
                cancel_id = (await call(session, "submit_embedding_job", **cancel_args))["job_id"]
                canceled = await call(session, "cancel_embedding_job", job_id=cancel_id)
                assert canceled["state"] in ("canceling", "canceled")
                await until(lambda: call(session, "embedding_job_status", job_id=cancel_id),
                            lambda r: r["state"] == "canceled")
                await call(session, "embedding_job_results", job_id=cancel_id, cursor=f"{job_id}:1", error="invalid_cursor")
            print(f"MCP resumed {items} items after an abrupt restart", flush=True)
        except BaseException:
            print(log_path.read_text()[-16000:], file=sys.stderr)
            raise
        finally:
            stop(process)

        process = start(ttl=2)
        try:
            async with mcp_client(url + "/mcp") as (session, _):
                expiring = (await call(session, "submit_embedding_job", model="minilm", inputs=["ttl"],
                                       idempotency_key="qa-expiry"))["job_id"]
                finished = await until(lambda: call(session, "embedding_job_status", job_id=expiring),
                                       lambda r: r["state"] == "completed")
                assert finished["expires_at"] - finished["finished"] == 2
                assert (await call(session, "embedding_job_results", job_id=expiring))["items"]
                await asyncio.sleep(2.1)
                await call(session, "embedding_job_results", job_id=expiring, error="job_not_found")
        finally:
            stop(process)
        return {
            "mcp_version": version, "items": items, "done_at_crash_lower_bound": status["done"],
            "replayed_page_sha256": digest(before), "result_sha256": digest(results),
            "default_ttl_seconds": 86400, "verified_short_ttl_seconds": 2, "canceled_job": True,
            "idle_seconds_in_test": 2, "gemma2_media_tokens": mixed_layout["token_count"],
        }


async def pressure_check(assets, release):
    original = runtime.available_mib
    memory = 8192
    runtime.available_mib = lambda: memory
    try:
        with tempfile.TemporaryDirectory(prefix="emb-pressure-") as directory:
            path = Path(directory)
            storage = Storage(path / "db", 60)
            try:
                worker = Worker(release, assets, path, 4, 300, 3072)
                async with Scheduler(worker, storage) as scheduler:
                    item = normalized_item("Memory pressure test")
                    assert "embedding" in (await scheduler.interactive("minilm", [item]))[0]
                    pid = worker.status()["pid"]
                    memory = 2048
                    scheduler.wake.set()

                    async def status():
                        return scheduler.status()

                    unloaded = await until(status, lambda r: r["phase"] == "unloaded", timeout=8)
                    assert unloaded["last_unload"]["reason"] == "memory_pressure"
                    error = (await scheduler.interactive("minilm", [item]))[0]["error"]
                    assert error["code"] == "memory_pressure"
                    job_id = storage.submit("pressure", "minilm", worker.identity("minilm", False),
                                            [item], "f32_base64")["job_id"]
                    await asyncio.sleep(1.2)
                    assert storage.status(job_id)["state"] == "queued"
                    memory = 8192
                    scheduler.wake.set()

                    async def job():
                        return storage.status(job_id)

                    await until(job, lambda r: r["state"] == "completed")
                    assert worker.status()["pid"] != pid
                    return {"injected_available_mib": [8192, 2048, 8192], "unload_reason": "memory_pressure",
                            "job_resumed": True}
            finally:
                storage.close()
    finally:
        runtime.available_mib = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "service", "all"))
    parser.add_argument("--worker", type=Path, default=ROOT / ".assets/embedding-worker/current")
    parser.add_argument("--assets", type=Path, default=Path(os.environ.get("EMBED_ASSET_CACHE", ROOT / ".assets")))
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 32 <= args.items <= 1000:
        parser.error("--items must be between 32 and 1000")
    release, assets = args.worker.resolve(), args.assets.resolve()
    manifest = json.loads((release / "manifest.json").read_text())
    evidence = {"engine": {key: manifest[key] for key in ("version", "artifact_id", "upstream_commit")}}
    if args.mode in ("native", "all"):
        evidence["native"] = native_checks(assets, release)
    if args.mode in ("service", "all"):
        evidence["service"] = asyncio.run(service_checks(assets, release, args.items))
        evidence["memory_pressure"] = asyncio.run(pressure_check(assets, release))
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(exist_ok=True, parents=True)
        args.out.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
