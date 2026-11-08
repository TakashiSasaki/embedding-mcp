import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

from common import ASSETS, PROJECT

LOCK = PROJECT / "assets.lock.json"
MANIFEST = json.loads(LOCK.read_text())
GROUPS = MANIFEST["groups"]
TEXT_MODELS = ("e5", "minilm", "labse", "gemma2", "qwen3")
REFERENCES = ("e5", "minilm", "labse")
ALIASES = {
    "text": ("runtime", *TEXT_MODELS),
    "multimodal": ("runtime", "gemma2", "gemma2-media"),
    "reference": ("runtime", *REFERENCES, *(f"ref-{model}" for model in REFERENCES)),
    "nonbert": ("ref-laser", "ref-potion"),
    "all": tuple(GROUPS),
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(path, asset):
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}")
    if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
        raise ValueError(f"Asset does not match assets.lock.json: {path}")


def check_platform():
    expected = MANIFEST["runtime"]["platform"]
    actual = f"{platform.system()}-{platform.machine()}"
    if actual != expected:
        raise ValueError(f"The pinned llama.cpp build requires {expected}, found {actual}")


def fetch_file(asset):
    target = ASSETS / asset["path"]
    if target.exists():
        verify(target, asset)
        print(f"Verified {asset['path']}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {asset['path']}", flush=True)
    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        temp = Path(handle.name)
    try:
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                        "--connect-timeout", "20", "--output", str(temp), asset["url"]], check=True)
        verify(temp, asset)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def server_version():
    binary = ASSETS / "llama-b12110/llama-server"
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True)
    version = (result.stdout + result.stderr).strip()
    commit = MANIFEST["runtime"]["commit"]
    if commit[:9] not in version:
        raise ValueError(f"Unexpected llama-server version at {binary}: {version}")
    return version


def install_runtime():
    target = ASSETS / "llama-b12110"
    if not target.exists():
        with TemporaryDirectory(dir=ASSETS) as stage:
            with tarfile.open(ASSETS / "llama-arm64.tar.gz") as archive:
                archive.extractall(stage, filter="data")
            try:
                (Path(stage) / target.name).rename(target)
            except OSError:
                if not target.is_dir():
                    raise
    print(server_version())


def require_assets(groups):
    if "runtime" in groups:
        check_platform()
    for group in groups:
        for asset in GROUPS[group]:
            verify(ASSETS / asset["path"], asset)
    return server_version() if "runtime" in groups else None


def main():
    parser = argparse.ArgumentParser(description="Fetch and verify pinned CPU experiment assets.")
    parser.add_argument("groups", nargs="+", choices=sorted(GROUPS.keys() | ALIASES.keys()))
    args = parser.parse_args()
    groups = dict.fromkeys(group for name in args.groups for group in ALIASES.get(name, (name,)))
    try:
        if "runtime" in groups:
            check_platform()
        for group in groups:
            for asset in GROUPS[group]:
                fetch_file(asset)
        if "runtime" in groups:
            install_runtime()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
