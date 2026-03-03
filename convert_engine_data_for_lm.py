#!/usr/bin/env python3
"""
Convert engine_data.jsonl (fen, uci) to prompt/completion format for LM fine-tuning.
Reads engine_data.jsonl only; does not modify generate_engine_data.py or the original file.

Output: JSONL with {"prompt": "FEN: <fen>\\nMove:\\n", "completion": "<uci>"}
Optional: --limit N to convert only first N lines (for testing).
Usage:
  python convert_engine_data_for_lm.py --out engine_data_lm.jsonl
  python convert_engine_data_for_lm.py --out engine_data_lm.jsonl --limit 100000
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Convert engine_data.jsonl to LM prompt/completion format.")
    p.add_argument("--input", default="engine_data.jsonl", help="Input JSONL (fen, uci) from generate_engine_data.py")
    p.add_argument("--out", default="engine_data_lm.jsonl", help="Output JSONL with prompt, completion")
    p.add_argument("--limit", type=int, default=None, help="Convert only first N lines (default: all)")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    count = 0
    with open(inp, "r", encoding="utf-8") as fr, open(args.out, "w", encoding="utf-8") as fw:
        for line in fr:
            if args.limit is not None and count >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                fen = d.get("fen")
                uci = d.get("uci")
                if not fen or not uci:
                    continue
                prompt = f"FEN: {fen}\nMove:\n"
                obj = {"prompt": prompt, "completion": uci}
                fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"Wrote {count} prompt/completion pairs to {args.out}")


if __name__ == "__main__":
    main()
