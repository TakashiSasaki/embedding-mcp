from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("EMBED_ASSET_CACHE", PROJECT / ".assets")).resolve()
FIXTURE = Path(os.environ.get("EMBED_FIXTURE", PROJECT / "fixtures/retrieval-ja-en-v1.json")).resolve()
LOCAL_HTTP = build_opener(ProxyHandler({}))


class Client:
    def __init__(self, port, pid):
        self.url = f"http://127.0.0.1:{port}"
        self.pid = pid

    def post(self, endpoint, payload):
        request = Request(self.url + endpoint, json.dumps(payload).encode(),
                          {"Content-Type": "application/json"})
        try:
            with LOCAL_HTTP.open(request, timeout=120) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"{endpoint}: HTTP {error.code}: {error.read().decode()}") from error


def peak_rss(pid):
    status = Path(f"/proc/{pid}/status").read_text()
    return round(int(next(line.split()[1] for line in status.splitlines()
                          if line.startswith("VmHWM:"))) / 1024)


@contextmanager
def server(model, log_path, *, threads, context, pooling=None, projector=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    binary = ASSETS / "llama-b12110/llama-server"
    args = [str(binary), "-m", str(model), "--embedding", "-ngl", "0", "-np", "1",
            "-t", str(threads), "-tb", str(threads), "-c", str(context), "-b", str(context),
            "-ub", str(context), "--host", "127.0.0.1", "--port", str(port)]
    if pooling:
        args += ["--pooling", pooling]
    if projector:
        args += ["--mmproj", str(projector), "--no-mmproj-offload", "--video-fps", "2"]
    else:
        args += ["--no-mmproj"]

    with log_path.open("w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        client = Client(port, process.pid)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited {process.returncode}. See {log_path}")
                try:
                    with LOCAL_HTTP.open(client.url + "/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                raise TimeoutError(f"llama-server did not become healthy. See {log_path}")
            yield client
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
