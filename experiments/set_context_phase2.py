"""Prepare or explicitly run the phase 2 CPU set-context protocol."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import time

from common import ASSETS, PROJECT
from fetch import GROUPS, LOCK, MANIFEST, sha256, verify
from run import git, make_output, write_json
from set_context import LANGUAGES, SOURCE, sense_margin, summarize_row

FIXTURE = PROJECT / "fixtures/set-context-ja-en-v2.json"
CONVERTER = Path(__file__).with_name("convert_phase2_mbert.py")
WORKER = ASSETS / "embedding-worker/current"
MODELS = ("minilm", "e5", "qwen3", "gemma2", "labse", "e5-base", "bge-m3", "mbert")
MODEL_MANIFEST = PROJECT / "fixtures/set-context-phase2-derived.json"
SPLITS = ("development", "heldout")
CONTEXT = 128
MAX_WEIGHT_BYTES = 650_000_000
MARGIN_TOLERANCE = 1e-6
LIBRARIES = ("libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so")


def now():
    return datetime.now(timezone.utc)


def read_json(path):
    return json.loads(path.read_text())


def log_stage(out, **row):
    with (out / "stages.jsonl").open("a") as log:
        log.write(json.dumps({"recorded_at": now().isoformat(), **row}) + "\n")


@contextmanager
def timed(out, stage, **details):
    started = now().isoformat()
    clock = time.perf_counter()
    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        log_stage(out, stage=stage, started_at=started, seconds=time.perf_counter() - clock,
                  status=status, **details)


def worker_manifest(worker):
    manifest = read_json(worker / "manifest.json")
    if manifest["upstream_commit"] != MANIFEST["runtime"]["commit"]:
        raise ValueError("Worker commit differs from asset lock")
    for name in LIBRARIES:
        if sha256(worker / name) != manifest["files"][name]:
            raise ValueError(f"Worker library differs from manifest: {name}")
    return manifest


def input_hashes():
    helpers = (SOURCE.with_name(name) for name in ("set_context.py", "fetch.py", "run.py", "common.py"))
    paths = (FIXTURE, LOCK, SOURCE, Path(__file__), CONVERTER, MODEL_MANIFEST, *helpers)
    return {str(path.relative_to(PROJECT)): sha256(path) for path in paths}


def case_counts(items):
    counts = Counter((item["split"], item["language"]) for item in items)
    return {split: {lang: counts[split, lang] for lang in LANGUAGES} for split in SPLITS}


def validate_fixture(fixture):
    if fixture["version"] != 2:
        raise ValueError("Expected phase 2 fixture")
    seen = set()
    for group in ("sense", "binding"):
        for item in fixture[group]:
            if item["id"] in seen or item["language"] not in LANGUAGES or item["split"] not in SPLITS:
                raise ValueError(f"Invalid or duplicate case: {item['id']}")
            seen.add(item["id"])
            if group == "sense":
                sides = [item[side] for side in ("a", "b")]
                sets = [[item["target"], *side["companions"]] for side in sides]
                if any(not side["reference"].strip() for side in sides) or any(len(row) != 3 for row in sets):
                    raise ValueError(f"Invalid sense case: {item['id']}")
            else:
                sets = [item[key] for key in ("original", "reassigned", "paraphrase")]
                if len(sets[0]) not in (2, 3) or len({len(row) for row in sets}) != 1 or sets[0] == sets[1]:
                    raise ValueError(f"Invalid binding case: {item['id']}")
            if any(len(set(row)) != len(row) or any(not text.strip() for text in row) for row in sets):
                raise ValueError(f"Empty or duplicate member: {item['id']}")


def verify_tokenization(report, fixture):
    if report["architecture"] not in {"bert", "qwen3", "gemma-embedding2"} or not 0 < report["layers"] < 64:
        raise ValueError("Unsupported architecture or layer count")
    for group in ("sense", "binding"):
        if [item["id"] for item in report[group]] != [item["id"] for item in fixture[group]]:
            raise ValueError(f"Preflight output is missing {group} cases")
    lengths = [length for item in report["binding"] for length in item["token_counts"]]
    for item in report["sense"]:
        for side in item["sides"]:
            if not side["anchor_token_ids_match_standalone"]:
                raise ValueError(f"Anchor target differs: {item['id']}")
            lengths.extend(side[key] for key in ("context_tokens", "definition_tokens", "legacy_tokens"))
    if max(lengths) > CONTEXT:
        raise ValueError(f"Input has {max(lengths)} tokens, over the {CONTEXT}-token limit")
    return max(lengths)


def verify_mbert_tokenizer(report):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ASSETS / "ref-mbert/tokenizer.json"))
    mismatches = [sample for sample, ids in report["tokenizations"].items()
                  if tokenizer.encode(sample).ids != ids]
    if mismatches:
        raise ValueError(f"mBERT token IDs differ for {len(mismatches)} inputs: {mismatches[:5]}")
    return len(report["tokenizations"])


def prepare(args, out):
    fixture = read_json(FIXTURE)
    validate_fixture(fixture)
    worker = args.worker.resolve()
    with timed(out, "verify_worker"):
        manifest = worker_manifest(worker)
    derived = read_json(MODEL_MANIFEST)
    if derived["worker_patch_sha256"] != manifest["patch_sha256"]:
        raise ValueError("Worker patch differs from the pinned mBERT conversion")
    source_tree = PROJECT / ".assets/mcp-build/sources" / manifest["patch_sha256"]
    binary = out / "set-context"
    build = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
             *(f"-I{source_tree / folder}" for folder in ("include", "ggml/include", "vendor")),
             str(SOURCE), f"-L{worker}", f"-Wl,-rpath,{worker}", "-lllama", "-o", str(binary)]
    with timed(out, "compile"), (out / "build.log").open("w") as log:
        subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    models = {}
    common_binding = {item["id"] for item in fixture["binding"]}
    for name in MODELS:
        asset = derived["gguf"] if name == "mbert" else GROUPS[name][0]
        if asset["bytes"] > MAX_WEIGHT_BYTES:
            raise ValueError(f"Weight exceeds CPU experiment budget: {name}")
        weights = ASSETS / asset["path"]
        with timed(out, "verify_weights", model=name):
            verify(weights, asset)
        report_path = out / f"tokenization-{name}.json"
        with timed(out, "tokenize_only", model=name), report_path.open("w") as stdout, \
                (out / f"tokenization-{name}.log").open("w") as stderr:
            subprocess.run([str(binary), str(weights), str(FIXTURE), "--tokenize-only"],
                           stdout=stdout, stderr=stderr, check=True, timeout=60)
        report = read_json(report_path)
        matched = {item["id"] for item in report["binding"] if item["matched"]}
        common_binding &= matched
        models[name] = {"asset": asset, "architecture": report["architecture"], "layers": report["layers"],
                        "max_input_tokens": verify_tokenization(report, fixture),
                        "matched_binding": case_counts(item for item in fixture["binding"] if item["id"] in matched),
                        "tokenization_sha256": sha256(report_path)}
        if name == "mbert":
            with timed(out, "verify_reference_tokenizer", model=name):
                for reference in GROUPS["ref-mbert"]:
                    if reference["path"].endswith("/tokenizer.json"):
                        verify(ASSETS / reference["path"], reference)
                models[name]["original_tokenizer_matched_inputs"] = verify_mbert_tokenizer(report)

    binding = [item for item in fixture["binding"] if item["id"] in common_binding]
    counts = case_counts(binding)
    if min(count for split in counts.values() for count in split.values()) < 4:
        raise ValueError(f"Fewer than 4 common binding cases in a split/language: {counts}")
    write_json(out / "plan.json", {
        "version": 2, "prepared_at": now().isoformat(), "inputs": input_hashes(), "worker": str(worker),
        "runtime": manifest, "binary_sha256": sha256(binary), "build": build,
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "host": platform.platform(), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")), "threads": 4, "context_tokens": CONTEXT,
        "models": models, "common_binding_ids": [item["id"] for item in binding],
        "counts": {"sense": case_counts(fixture["sense"]), "binding_candidates": case_counts(fixture["binding"]),
                   "common_binding": counts},
    })


def binding_margin(item):
    return min(item["ranking_margin"], item["ranking_margin_reverse"])


def aggregate(row, report, language=None):
    result = summarize_row(row, language)
    binding = [item for item in row["binding"] if language in (None, item["language"])]
    equal_context = {item["id"] for item in report["sense"]
                     if item["sides"][0]["context_tokens"] == item["sides"][1]["context_tokens"]}
    matched = [item for item in row["sense"] if item["id"] in equal_context
               and language in (None, item["language"])]
    sense = [item for item in row["sense"] if language in (None, item["language"])]
    result.update(sense_both_correct_above_tolerance=sum(sense_margin(item) > MARGIN_TOLERANCE for item in sense),
                  binding_both_correct=sum(binding_margin(item) > MARGIN_TOLERANCE for item in binding),
                  binding_median_min_margin=median(map(binding_margin, binding)),
                  equal_context_sense_cases=len(matched),
                  equal_context_sense_both_correct=sum(sense_margin(item) > MARGIN_TOLERANCE for item in matched),
                  equal_context_sense_median_min_margin=median(map(sense_margin, matched)) if matched else None)
    return result


def select_boundaries(rows):
    metrics = (("sense", "sense_both_correct_above_tolerance", "sense_median_min_margin"),
               ("binding", "binding_both_correct", "binding_median_min_margin"))
    return {task: max(rows, key=lambda row: (row["all"][correct], row["all"][margin], -row["k"]))["k"]
            for task, correct, margin in metrics}


def run(args, out):
    prepared = args.prepared.resolve()
    plan_path = prepared / "plan.json"
    plan = read_json(plan_path)
    plan_hash = sha256(plan_path)
    model = plan["models"][args.model]
    worker = Path(plan["worker"])
    binary = prepared / "set-context"
    weights = ASSETS / model["asset"]["path"]
    report_path = prepared / f"tokenization-{args.model}.json"
    with timed(out, "verify_inputs"):
        if (plan["version"] != 2 or input_hashes() != plan["inputs"] or
                worker_manifest(worker) != plan["runtime"] or
                sha256(binary) != plan["binary_sha256"] or sha256(report_path) != model["tokenization_sha256"]):
            raise ValueError("Preparation inputs have changed. Rerun prepare.")
        verify(weights, model["asset"])

    full_sweep = list(range(model["layers"] + 1))
    boundaries = full_sweep
    decisions = None
    development_hash = None
    if args.split == "heldout":
        dev_path = args.development / "summary.json"
        development = read_json(dev_path)
        if (development["plan_sha256"] != plan_hash or development["model"] != args.model or
                development["split"] != "development" or
                [row["k"] for row in development["boundaries"]] != full_sweep):
            raise ValueError("Held-out run requires a complete development sweep for this model and plan")
        decisions = select_boundaries(development["boundaries"])
        boundaries = sorted({0, model["layers"], *decisions.values()})
        development_hash = sha256(dev_path)

    original = read_json(FIXTURE)
    case_ids = set(plan["common_binding_ids"])
    fixture = {"version": 2, "boundaries": boundaries,
               "sense": [item for item in original["sense"] if item["split"] == args.split],
               "binding": [item for item in original["binding"]
                           if item["split"] == args.split and item["id"] in case_ids]}
    run_fixture = out / "fixture.json"
    write_json(run_fixture, fixture)
    metadata = {"plan_sha256": plan_hash, "model": args.model, "split": args.split,
                "requested_boundaries": boundaries, "selected_on_development": decisions,
                "margin_tolerance": MARGIN_TOLERANCE,
                "development_summary_sha256": development_hash, "fixture_sha256": sha256(run_fixture)}
    write_json(out / "metadata.json", metadata)
    raw_path = out / "raw.jsonl"
    with timed(out, "worker_total", model=args.model, split=args.split), raw_path.open("w") as raw, \
            (out / "worker.log").open("w") as log:
        with subprocess.Popen([str(binary), str(weights), str(run_fixture)], stdout=raw, stderr=subprocess.PIPE,
                              text=True) as process:
            for line in process.stderr:
                log.write(line)
                if line.startswith("PHASE2_TIMING "):
                    stage = json.loads(line.removeprefix("PHASE2_TIMING "))
                    log_stage(out, **stage, status="complete", model=args.model, split=args.split)
            if process.wait():
                raise RuntimeError(f"Worker exited {process.returncode}. See {out / 'worker.log'}")

    with timed(out, "aggregate"):
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        if [row["k"] for row in rows] != boundaries:
            raise ValueError("Incomplete or unexpected boundary sweep")
        report = read_json(report_path)
        aggregates = []
        for row in rows:
            for group in ("sense", "binding"):
                if [item["id"] for item in row[group]] != [item["id"] for item in fixture[group]]:
                    raise ValueError(f"Missing {group} cases at k={row['k']}")
            aggregates.append({"k": row["k"], "all": aggregate(row, report),
                               **{lang: aggregate(row, report, lang) for lang in LANGUAGES}})
        write_json(out / "summary.json", {**metadata, "boundaries": aggregates})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    preparation = sub.add_parser("prepare", help="Compile and check token IDs without inference")
    preparation.add_argument("--worker", type=Path, default=WORKER)
    execution = sub.add_parser("run", help="Start embedding inference for one model and split")
    execution.add_argument("--prepared", type=Path, required=True)
    execution.add_argument("--model", choices=MODELS, required=True)
    execution.add_argument("--split", choices=SPLITS, required=True)
    execution.add_argument("--development", type=Path)
    for command in (preparation, execution):
        command.add_argument("--out", type=Path, help="New output directory, default under runs/")
    args = parser.parse_args()
    if args.action == "run" and (args.split == "heldout") != bool(args.development):
        parser.error("--development is required for heldout only")
    out = make_output(args.out, f"set-context-phase2-{args.action}", now())
    print(f"Output: {out}", flush=True)
    with timed(out, args.action):
        if args.action == "prepare":
            prepare(args, out)
        else:
            run(args, out)


if __name__ == "__main__":
    main()
