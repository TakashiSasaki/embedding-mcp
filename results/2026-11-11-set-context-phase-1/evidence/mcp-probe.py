import asyncio
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import httpx

sys.path.insert(0, "/data/cpu-embedding-experiments/main/service")
from contracts import MODELS, ROOT, digest
from verify import call, choose_port, mcp_client, stop, wait_http

A = ["白い猫", "黒い犬"]
B = ["白い犬", "黒い猫"]

def catalog(tools):
    return [tool.model_dump(by_alias=True, mode="json", exclude_none=True)
            for tool in sorted(tools, key=lambda tool: tool.name) if tool.name != "embed_set"]

async def probe(url):
    async with mcp_client("http://127.0.0.1:18765/mcp") as (live, _):
        existing = catalog((await live.list_tools()).tools)
    async with mcp_client(url + "/mcp") as (session, version):
        tools = (await session.list_tools()).tools
        assert catalog(tools) == existing
        new = next(tool for tool in tools if tool.name == "embed_set")
        assert set(new.inputSchema["required"]) == {"model", "inputs"}
        rows = []
        for model, spec in MODELS.items():
            prefix = "passage: " if model == "e5" else ""
            inputs_a = [prefix + item for item in A]
            inputs_b = [prefix + item for item in B]
            original = await call(session, "embed_set", model=model, inputs=inputs_a)
            reversed_a = await call(session, "embed_set", model=model, inputs=inputs_a[::-1])
            reassigned = await call(session, "embed_set", model=model, inputs=inputs_b)
            assert original == reversed_a
            assert original["set_id"] == digest(sorted(inputs_a))
            assert reassigned["set_id"] == digest(sorted(inputs_b)) != original["set_id"]
            a, b = original["embedding"], reassigned["embedding"]
            assert len(a) == len(b) == spec["dimension"]
            norm_a, norm_b = math.hypot(*a), math.hypot(*b)
            assert abs(norm_a - 1) < 1e-6 and abs(norm_b - 1) < 1e-6
            delta = max(abs(x - y) for x, y in zip(a, b, strict=True))
            assert delta > 0
            rows.append({
                "model": model, "inputs_a": inputs_a, "inputs_b": inputs_b,
                "dimension": len(a), "same_under_reversal": True, "set_id_a": original["set_id"],
                "set_id_b": reassigned["set_id"], "max_abs_delta": delta,
                "cosine": math.fsum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b),
                "model_sha256": original["model_sha256"],
            })
        return {"mcp_version": version, "mcp_catalog_count": len(tools),
                "unchanged_catalog_sha256": hashlib.sha256(json.dumps(
                    existing, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "threads": 4, "results": rows}

def main():
    run_dir = Path(__file__).parent
    worker = (ROOT / ".assets/embedding-worker/current").resolve()
    with TemporaryDirectory(prefix="embed-set-pair-") as directory:
        tmp = Path(directory)
        port = choose_port()
        url = f"http://127.0.0.1:{port}"
        with (run_dir / "service.log").open("w") as log:
            process = subprocess.Popen([
                sys.executable, str(ROOT / "service/main.py"), "--worker", str(worker),
                "--assets", str(ROOT / ".assets"), "--state-dir", str(tmp / "state"),
                "--runtime-dir", str(tmp / "runtime"), "--port", str(port), "--threads", "4",
            ], stdout=log, stderr=log, start_new_session=True)
            try:
                with httpx.Client(base_url=url, trust_env=False) as client:
                    wait_http(client, process, "/health")
                result = asyncio.run(probe(url))
                result["worker"] = worker.name
                result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                result["service_source_sha256"] = hashlib.sha256((ROOT / "service/main.py").read_bytes()).hexdigest()
                (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(run_dir / "summary.json")
                for row in result["results"]:
                    print(f'{row["model"]} {row["dimension"]} {row["cosine"]:.9f} {row["max_abs_delta"]:.9f}')
                print(f'Unchanged catalog SHA-256: {result["unchanged_catalog_sha256"]}')
            finally:
                stop(process)

if __name__ == "__main__":
    main()
