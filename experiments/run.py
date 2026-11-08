import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from tempfile import mkdtemp

from common import FIXTURE, PROJECT
from fetch import LOCK, REFERENCES, TEXT_MODELS, require_assets, sha256


def git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT, text=True,
                          capture_output=True).stdout.strip()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def save(out, name, result, summaries):
    write_json(out / f"{name}.json", result)
    summaries[name] = {k: v for k, v in result.items() if k not in ("vectors", "tokens")}
    write_json(out / "summary.json", summaries)


def make_output(path, suite, started):
    if path:
        path = path.resolve()
        path.mkdir(parents=True)
        return path
    root = PROJECT / "runs"
    root.mkdir(exist_ok=True)
    return Path(mkdtemp(prefix=f"{started:%Y%m%dT%H%M%SZ}-{suite}-", dir=root))


def main():
    parser = argparse.ArgumentParser(description="Run a pinned CPU embedding experiment.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, help="New directory for this run. Default is an isolated directory under runs/.")
    suites = parser.add_subparsers(dest="suite", required=True)
    text = suites.add_parser("text", help="Multilingual text retrieval with llama.cpp.")
    text.add_argument("models", nargs="*", metavar="MODEL", help=", ".join(TEXT_MODELS))
    ref = suites.add_parser("reference", help="Transformers parity, MiniLM mask, and LaBSE diagnosis.")
    ref.add_argument("models", nargs="*", metavar="MODEL", help=", ".join(REFERENCES))
    local = suites.add_parser("nonbert", help="Run one in-process non-BERT benchmark.")
    local.add_argument("model", choices=("laser2", "potion"))
    suites.add_parser("multimodal", help="EmbeddingGemma 2 with synthetic text, image, audio, and video.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    groups = []
    if args.suite in ("text", "reference"):
        choices = TEXT_MODELS if args.suite == "text" else REFERENCES
        models = tuple(dict.fromkeys(args.models or choices))
        if unknown := set(models) - set(choices):
            parser.error(f"Unknown {args.suite} model: {', '.join(sorted(unknown))}")
        groups = ["runtime", *models]
        if args.suite == "reference":
            groups += [f"ref-{model}" for model in models]
    elif args.suite == "multimodal":
        groups = ["runtime", "gemma2", "gemma2-media"]
    else:
        groups = ["ref-laser" if args.model == "laser2" else "ref-potion"]

    try:
        llama_version = require_assets(groups)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\nRun: python3 experiments/fetch.py {' '.join(groups)}\n")

    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      OPENBLAS_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads))
    dependencies = {}
    if args.suite == "reference":
        dependencies = {name: version(name) for name in ("torch", "transformers", "safetensors")}
    elif args.suite == "nonbert":
        names = ("torch", "sentencepiece", "sacremoses", "unicategories") if args.model == "laser2" else ("model2vec", "numpy")
        dependencies = {name: version(name) for name in names}
    if "torch" in dependencies:
        import torch

        if torch.version.cuda is not None:
            parser.error("Use the locked CPU-only PyTorch environment")
        torch.set_num_interop_threads(1)

    started = datetime.now(timezone.utc)
    out = make_output(args.out, args.suite, started)
    meta = {
        "started_at": started.isoformat(),
        "command": shlex.join(sys.argv),
        "git_commit": git("rev-parse", "--verify", "HEAD") or None,
        "git_dirty": bool(git("status", "--porcelain")),
        "asset_lock_sha256": sha256(LOCK),
        "python_lock_sha256": sha256(PROJECT / "uv.lock") if dependencies else None,
        "asset_groups": groups,
        "fixture": None if args.suite == "multimodal" else FIXTURE.name,
        "fixture_sha256": None if args.suite == "multimodal" else sha256(FIXTURE),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": dependencies,
        "threads": args.threads,
        "llama_cpp": llama_version,
    }
    if args.suite == "multimodal":
        meta["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    write_json(out / "metadata.json", meta)
    print(f"Run directory: {out}", flush=True)
    summaries = {}

    if args.suite == "multimodal":
        from multimodal import benchmark

        save(out, "multimodal", benchmark(out, args.threads), summaries)
    elif args.suite == "nonbert":
        from nonbert import benchmark

        save(out, args.model, benchmark(args.model, args.threads), summaries)
    else:
        from text import run_gguf, run_reference

        for model in models:
            stock = run_gguf(model, out, args.threads)
            save(out, f"{model}-gguf", stock, summaries)
            if args.suite == "reference":
                reference = run_reference(model, stock, args.threads)
                save(out, f"{model}-reference", reference, summaries)
                if model == "labse":
                    from labse import diagnose

                    save(out, "labse-diagnosis", diagnose(stock, reference, out, args.threads), summaries)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "metadata.json", meta)
    print(f"Summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
