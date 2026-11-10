"""Sweep the first open attention layer in the pinned CPU MiniLM worker."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
from statistics import median
import subprocess
import sys

from common import ASSETS, PROJECT
from fetch import GROUPS, LOCK, MANIFEST, sha256, verify
from run import git, make_output, write_json

SOURCE = Path(__file__).with_suffix(".cpp")
FIXTURE = PROJECT / "fixtures/set-context-ja-en-v1.json"
WORKER = PROJECT / ".assets/embedding-worker/current"
LANGUAGES = ("ja", "en")


def sense_margin(item, anchor="definition"):
    return min(item[anchor]["margins"])


def summarize_row(row, language=None):
    sense = [item for item in row["sense"] if language in (None, item["language"])]
    binding = [item for item in row["binding"] if language in (None, item["language"])]
    return {
        "sense_cases": len(sense),
        "sense_both_correct": sum(sense_margin(item) > 0 for item in sense),
        "sense_median_min_margin": median(map(sense_margin, sense)),
        "legacy_both_correct": sum(sense_margin(item, "legacy") > 0 for item in sense),
        "legacy_median_min_margin": median(sense_margin(item, "legacy") for item in sense),
        "context_distance_median": median(item["context_distance"] for item in sense),
        "binding_cases": len(binding),
        "binding_rank_correct": sum(item["ranking_margin"] > 0 for item in binding),
        "binding_margin_median": median(item["ranking_margin"] for item in binding),
        "swap_distance_median": median(item["swap_distance"] for item in binding),
        "token_mean_swap_distance_median": median(item["token_mean_swap_distance"] for item in binding),
        "swap_exceeds_raw_reverse": sum(item["swap_distance"] > item["raw_reverse_set_distance"] for item in binding),
        "raw_reverse_set_distance_max": max(item["raw_reverse_set_distance"] for item in sense + binding),
        "orders_checked": sum(item["orders_checked"] for item in sense + binding),
    }


def summarize(rows, fixture):
    if [row["k"] for row in rows] != list(range(13)):
        raise ValueError("Expected one result for each of k=0 through 12")
    for row in rows:
        for group in ("sense", "binding"):
            if [item["id"] for item in row[group]] != [item["id"] for item in fixture[group]]:
                raise ValueError(f"Wrong cases in {group} at k={row['k']}")
    aggregates = [{"k": row["k"], "all": summarize_row(row),
                   **{lang: summarize_row(row, lang) for lang in LANGUAGES}} for row in rows]
    adjacent = []
    for old, new in zip(rows, rows[1:]):
        # This moves just layer k from open to blocked.
        sense = zip(old["sense"], new["sense"], strict=True)
        binding = list(zip(old["binding"], new["binding"], strict=True))
        adjacent.append({
            "newly_blocked_layer": old["k"],
            "sense_min_margin_change_median": median(sense_margin(b) - sense_margin(a) for a, b in sense),
            "binding_rank_margin_change_median": median(b["ranking_margin"] - a["ranking_margin"] for a, b in binding),
            "swap_distance_change_median": median(b["swap_distance"] - a["swap_distance"] for a, b in binding),
        })
    return {
        "boundaries": aggregates, "adjacent": adjacent,
        "permutations": sum(row["all"]["orders_checked"] for row in aggregates),
        "max_raw_reverse_set_distance": max(row["all"]["raw_reverse_set_distance_max"] for row in aggregates),
    }


def plot(summary, path):
    import matplotlib.pyplot as plt

    rows = summary["boundaries"]
    k = [row["k"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, layout="constrained")
    specs = (
        ("sense_both_correct", "Both senses ranked correctly", "Number of words"),
        ("sense_median_min_margin", "Margin for the weaker of the two senses", "Median cosine margin"),
        ("swap_distance_median", "Same tokens, different pairings", "Median cosine distance"),
        ("binding_margin_median", "Paraphrase ranked against the swapped set", "Median cosine margin"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, specs, strict=True):
        for lang, label, color in (("ja", "Japanese", "#005c9e"), ("en", "English", "#b64a12")):
            axis.plot(k, [row[lang][key] for row in rows], marker="o", ms=4, color=color, label=label)
        axis.axvline(6, color="#999999", linestyle=":", linewidth=1)
        axis.set(title=title, ylabel=ylabel, xticks=k)
        axis.grid(axis="y", color="#e3e3e3", linewidth=.8)
    axes[0, 0].legend(frameon=False, loc="upper right")
    max_words = max(rows[0][lang]["sense_cases"] for lang in LANGUAGES)
    axes[0, 0].set(yticks=list(range(max_words + 1)), ylim=(-.25, max_words + .45))
    axes[0, 1].axhline(0, color="#666666", linewidth=.8)
    axes[1, 0].set_yscale("log")
    noise = summary["max_raw_reverse_set_distance"]
    if noise > 0:
        axes[1, 0].axhline(noise, color="#666666", linestyle="--", linewidth=1, label="Largest raw reorder distance")
        axes[1, 0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[1, 1].axhline(0, color="#666666", linewidth=.8)
    for axis in axes[1]:
        axis.set_xlabel("Number of initial layers with cross-member attention blocked")
    fig.suptitle("MiniLM attention boundary sweep (12 layers)", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--worker", type=Path, default=WORKER)
    parser.add_argument("--out", type=Path, help="New run directory, default under runs/")
    parser.add_argument("--plot", action="store_true", help="Save a PNG with matplotlib")
    args = parser.parse_args()
    if args.plot:
        try:
            import matplotlib
        except ImportError:
            parser.error("--plot needs matplotlib")

    worker = args.worker.resolve()
    manifest = json.loads((worker / "manifest.json").read_text())
    if manifest["upstream_commit"] != MANIFEST["runtime"]["commit"]:
        raise ValueError("Worker upstream does not match assets.lock.json")
    for name in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so"):
        if sha256(worker / name) != manifest["files"][name]:
            raise ValueError(f"Worker library does not match the manifest: {name}")
    source_tree = PROJECT / ".assets/mcp-build/sources" / manifest["patch_sha256"]
    if not (source_tree / "include/llama.h").is_file():
        raise FileNotFoundError(f"Missing headers for {worker}; run service/build_worker.py to prepare a worker")
    asset = GROUPS["minilm"][0]
    weights = ASSETS / asset["path"]
    verify(weights, asset)
    fixture_path = args.fixture.resolve()
    fixture = json.loads(fixture_path.read_text())
    if fixture["version"] != 1:
        raise ValueError("Expected fixture version 1")
    for group in ("sense", "binding"):
        if {item["language"] for item in fixture[group]} != set(LANGUAGES):
            raise ValueError(f"Expected Japanese and English cases in {group}")
    started = datetime.now(timezone.utc)
    if args.out:
        out = args.out.resolve()
        out.mkdir(parents=True)
    else:
        out = make_output(None, "minilm-boundary", started)
    binary = out / "set-context"
    build = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", f"-I{source_tree / 'include'}",
             f"-I{source_tree / 'ggml/include'}", f"-I{source_tree / 'vendor'}", str(SOURCE),
             f"-L{worker}", f"-Wl,-rpath,{worker}", "-lllama", "-o", str(binary)]
    metadata = {
        "started_at": started.isoformat(), "command": shlex.join(["python3", *sys.argv]),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "threads": 4, "context_tokens": 128, "platform": platform.platform(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "asset_lock_sha256": sha256(LOCK), "model": asset,
        "fixture": fixture_path.name, "fixture_sha256": sha256(fixture_path),
        "source_sha256": sha256(SOURCE), "runner_sha256": sha256(Path(__file__)),
        "runtime": {key: manifest[key] for key in ("version", "artifact_id", "upstream_commit", "patch_sha256")},
        "anchors": {"primary": "target : definition, only target tokens pooled",
                    "sensitivity_check": "target and definition as two members, k=6 for all conditions"},
    }
    if args.plot:
        metadata["matplotlib"] = matplotlib.__version__
    write_json(out / "metadata.json", metadata)
    print(f"Run directory: {out}", flush=True)
    subprocess.run(build, check=True, timeout=120)
    with (out / "raw.jsonl").open("w") as raw:
        subprocess.run([str(binary), str(weights), str(fixture_path)], stdout=raw, check=True, timeout=480)
    rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
    summary = summarize(rows, fixture)
    write_json(out / "summary.json", summary)
    if args.plot:
        plot(summary, out / "boundary-sweep.png")


if __name__ == "__main__":
    main()
