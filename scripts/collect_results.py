"""Collect chrF++ (English -> target, test split) into a Table-1-like markdown table.

Usage: uv run python scripts/collect_results.py [--results-dir results]
Columns: Base (vanilla NLLB, evaluated before training), BT, UMT, RT, Ours.
"""
import argparse
import json
import re
from pathlib import Path

LANGS = ["ayr_Latn", "fur_Latn", "wol_Latn", "bho_Deva", "dyu_Latn", "rus_Cyrl"]
COLS = ["base", "bt", "umt", "rt", "ours"]
PATTERN = re.compile(r"chrF\+\+ \(src.tgt\):\s*([0-9.]+)")


def ours(run_dir):
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return None, None
    tests = [json.loads(l)["test/chrf++"] for l in f.read_text(encoding="utf-8").splitlines()
             if '"split": "test"' in l]
    if not tests:
        return None, None
    return tests[0], (tests[-1] if len(tests) > 1 else None)


def baseline(run_dir):
    f = run_dir / "log.txt"
    if not f.exists():
        return None
    m = PATTERN.findall(f.read_text(encoding="utf-8", errors="ignore"))
    return float(m[-1]) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    root = Path(args.results_dir)

    for model in ("600M", "1.3B"):
        rows = {}
        for lang in LANGS:
            base, final = ours(root / "ours" / model / lang)
            rows[lang] = {"base": base, "ours": final,
                          **{m: baseline(root / m / model / lang) for m in ("bt", "umt", "rt")}}
        print(f"\n### NLLB-distilled-{model} (test chrF++, eng -> target)\n")
        print("| lang | " + " | ".join(COLS) + " |")
        print("|---|" + "---|" * len(COLS))
        for lang in LANGS:
            cells = ["-" if rows[lang][c] is None else f"{rows[lang][c]:.2f}" for c in COLS]
            print(f"| {lang} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
