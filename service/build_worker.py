import argparse
import fcntl
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "service/native"
UPSTREAM = json.loads((NATIVE / "upstream.json").read_text())
RUNTIME = json.loads((ROOT / "assets.lock.json").read_text())["runtime"]
CACHE = ROOT / ".assets/mcp-build"
RELEASES = ROOT / ".assets/embedding-worker"
BUILD = CACHE / "build"


def sha_file(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def checked(*command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def atomic_link(source, target):
    temp = target.with_name(f".{target.name}.{os.getpid()}")
    temp.symlink_to(source)
    try:
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def archive_path():
    archive = CACHE / "upstream.tar.gz"
    if not archive.exists():
        with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as download:
            stage = Path(download.name)
        try:
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
                            "--output", str(stage), UPSTREAM["url"]], check=True)
            if sha_file(stage) != UPSTREAM["sha256"]:
                raise ValueError("Upstream archive SHA-256 mismatch")
            stage.replace(archive)
        finally:
            stage.unlink(missing_ok=True)
    if sha_file(archive) != UPSTREAM["sha256"]:
        raise ValueError("Cached upstream archive SHA-256 mismatch")
    return archive


def source_tree(archive, patches, patch_id):
    source = CACHE / "sources" / patch_id
    if not source.exists():
        source.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
            with tarfile.open(archive) as tar:
                tar.extractall(tmp, filter="data")
            tree = Path(tmp) / f"llama.cpp-{RUNTIME['commit']}"
            env = {**os.environ, "GIT_CEILING_DIRECTORIES": tmp}
            for patch in patches:
                subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=tree, env=env, check=True)
            tree.rename(source)
    return source


def publish(binary, revision, dirty, patch_id):
    files = sorted([binary, *binary.parent.glob("*.so*")])
    content = {file.name: sha_file(file) for file in files}
    identity = sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    suffix = revision[:12] + (".dirty" if dirty else "")
    engine = f"0.9.0-dev+local.{suffix}"
    output = subprocess.check_output([str(binary), "--version"], stderr=subprocess.STDOUT, text=True)
    if f"version: {engine} " not in output or RUNTIME["commit"][:9] not in output:
        raise RuntimeError(f"Unexpected worker version: {output.strip()}")
    RELEASES.mkdir(exist_ok=True)
    target = RELEASES / f"{suffix}-{identity[:16]}"
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=RELEASES) as tmp:
            directory = Path(tmp) / "release"
            directory.mkdir()
            for file in files:
                shutil.copy2(file, directory / file.name, follow_symlinks=False)
            manifest = {
                "version": engine, "artifact_id": identity, "files": content,
                "upstream_release": RUNTIME["release"], "upstream_commit": RUNTIME["commit"],
                "upstream_archive_sha256": UPSTREAM["sha256"], "patch_sha256": patch_id,
                "source_revision": revision, "dirty": dirty,
            }
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            directory.rename(target)
    atomic_link(Path(target.name), RELEASES / "current")
    print(f"{engine} {identity}\n{target}")


def main():
    parser = argparse.ArgumentParser(description="Build the pinned CPU llama.cpp worker and publish an immutable local release.")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 4))
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        revision = checked("git", "rev-parse", "HEAD", cwd=ROOT)
        dirty = bool(checked("git", "status", "--porcelain", cwd=ROOT))
        patches = sorted(NATIVE.glob("*.patch"))
        patch_id = sha256(b"".join(patch.read_bytes() for patch in patches)).hexdigest()
        source = source_tree(archive_path(), patches, patch_id)
        # Reuse object files when only the experiment repository identity changes.
        build = BUILD.with_name(f"build-{patch_id[:16]}")
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(CACHE)}
        suffix = revision[:12] + (".dirty" if dirty else "")
        subprocess.run([
            "cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_NUMBER=12110", f"-DLLAMA_BUILD_COMMIT={RUNTIME['commit'][:9]}",
            f"-DLLAMA_LOCAL_BUILD={suffix}", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_OPENSSL=OFF",
            "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF", "-DGGML_HIP=OFF",
        ], env=env, check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "-j", str(args.jobs)],
                       env=env, check=True)
        if revision != checked("git", "rev-parse", "HEAD", cwd=ROOT) or patch_id != sha256(
                b"".join(patch.read_bytes() for patch in patches)).hexdigest():
            raise RuntimeError("The source changed during the build. Re-run against the current revision.")
        publish(build / "bin/llama-server", revision, dirty, patch_id)


if __name__ == "__main__":
    main()
