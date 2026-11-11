"""Convert the pinned cased mBERT encoder using the phase 2 worker source."""

import argparse
import json
from pathlib import Path
import runpy
import sys

from common import ASSETS, PROJECT
from fetch import GROUPS, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=ASSETS / "embedding-worker/current")
    parser.add_argument("--out", type=Path, default=ASSETS / "mbert-base-cased-Q8_0.gguf")
    args = parser.parse_args()
    for asset in GROUPS["ref-mbert"]:
        verify(ASSETS / asset["path"], asset)
    reference = ASSETS / "ref-mbert"
    tokenizer = json.loads((reference / "tokenizer.json").read_text())
    if tokenizer["pre_tokenizer"]["type"] != "BertPreTokenizer" or "vocab" not in tokenizer["model"]:
        raise ValueError("Expected the pinned cased BERT WordPiece tokenizer")
    manifest = json.loads((args.worker / "manifest.json").read_text())
    source = PROJECT / ".assets/mcp-build/sources" / manifest["patch_sha256"]
    sys.path[:0] = [str(source), str(source / "gguf-py")]
    from conversion.bert import BertModel

    # The converter's hash table identifies BPE pre-tokenizers. The pinned
    # llama.cpp uses the default pre-tokenizer for WordPiece regardless of
    # this GGUF field. Keep the converter's token IDs and special tokens.
    BertModel.get_vocab_base_pre = lambda self, tokenizer: "default"
    converter = source / "convert_hf_to_gguf.py"
    sys.argv = [str(converter), str(reference), "--outfile", str(args.out), "--outtype", "q8_0"]
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
