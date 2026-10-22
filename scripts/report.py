"""Summarise runs produced by scripts/run_stage.py.

    python scripts/report.py results/pilot/600M/ayr_Latn/seed27 [more run dirs ...]
    python scripts/report.py --table results        # one line per run found under results/

For every run it prints:
  * test chrF++ before / after (metrics.jsonl), with the paper's Base/Ours value for reference,
  * a paired bootstrap significance test before vs after (sacrebleu, needs hyps/ from main.py),
  * training dynamics from train_log.jsonl: mean forward chrF (against the real reference, monitoring only),
    mean backward reward (the RL objective) and KL in the first vs last 10% of steps, speed and peak GPU memory.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# Table 1 of the paper (chrF++, eng -> tgt, test) as extracted from arxiv.org/html/2601.12535.
# Verify against the PDF before quoting. columns: base, bt, umt, rt, ours
PAPER = {
    "600M": {
        "ayr_Latn": (24.38, 27.34, 24.85, 26.68, 28.13), "wol_Latn": (20.73, 23.57, 19.89, 22.84, 23.96),
        "rus_Cyrl": (48.80, 53.31, 51.70, 51.63, 52.79), "fur_Latn": (45.75, 47.38, 48.96, 46.25, 47.61),
        "dyu_Latn": (18.03, 20.13, 20.82, 21.89, 22.50), "bho_Deva": (43.07, 45.36, 39.79, 41.62, 43.97),
    },
    "1.3B": {
        "ayr_Latn": (26.01, 28.61, 28.21, 27.59, 28.73), "wol_Latn": (24.73, 26.35, 27.72, 25.21, 26.66),
        "rus_Cyrl": (50.10, 55.46, 55.14, 54.07, 55.16), "fur_Latn": (49.41, 51.72, 46.43, 49.68, 50.76),
        "dyu_Latn": (18.24, 20.89, 22.93, 21.96, 22.97), "bho_Deva": (44.31, 47.23, 44.43, 44.52, 44.77),
    },
}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_run_dir(run):
    # results/<stage>/<model>/<lang>/<name>
    parts = run.resolve().parts
    try:
        i = parts.index("results")
        return parts[i + 1], parts[i + 2], parts[i + 3]
    except (ValueError, IndexError):
        return None, None, None


def test_scores(run):
    tests = [m for m in read_jsonl(run / "metrics.jsonl") if m["split"] == "test"]
    before = tests[0]["test/chrf++"] if tests else None
    after = tests[-1]["test/chrf++"] if len(tests) > 1 else None
    n = tests[0].get("n") if tests else None
    return before, after, n


def window_mean(rows, key, frac, last):
    k = max(1, int(len(rows) * frac))
    sel = rows[-k:] if last else rows[:k]
    return statistics.fmean(r[key] for r in sel)


def paired_bootstrap(run):
    """sacrebleu paired bootstrap (1000 resamples), run inside hyps/ so that the table shows short file names."""
    hyp_dir = run / "hyps"
    names = ("test.ref.txt", "test_before.hyp.txt", "test_after.hyp.txt")
    if not all((hyp_dir / n).exists() for n in names):
        return None
    if (hyp_dir / names[1]).read_bytes() == (hyp_dir / names[2]).read_bytes():
        # sacrebleu reports a spurious "significant" p for identical systems
        return "before/after translations are identical: no change at all (nothing to test)"
    cmd = [sys.executable, "-m", "sacrebleu", names[0], "-i", names[1], names[2], "-m", "chrf",
           "--chrf-word-order", "2", "--paired-bs", "--paired-bs-n", "1000", "-f", "text"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=hyp_dir)
    lines = []
    for line in out.stdout.splitlines():
        if line.startswith("----"):  # start of sacrebleu's long explanation
            break
        lines.append(line)
    return chr(10).join(lines).strip()


def report(run, bootstrap=True):
    run = Path(run)
    stage, model, lang = parse_run_dir(run)
    print(f"\n=== {run}")
    before, after, n = test_scores(run)
    paper = PAPER.get(model, {}).get(lang)
    if before is not None:
        print(f"test chrF++ before: {before:.2f}   (n={n} sentences)" +
              (f"   [paper Base: {paper[0]:.2f}]" if paper else ""))
    if after is not None:
        print(f"test chrF++ after : {after:.2f}   delta = {after - before:+.2f}" +
              (f"   [paper Ours: {paper[4]:.2f}, paper delta = {paper[4] - paper[0]:+.2f}]" if paper else ""))
    evals = [m for m in read_jsonl(run / "metrics.jsonl") if m["split"] == "eval"]
    if len(evals) > 1:
        print("val chrF++ by step: " + ", ".join(f"{m['step']}:{m['eval/chrf++']:.2f}" for m in evals))

    rows = read_jsonl(run / "train_log.jsonl")
    if rows:
        print(f"training steps logged: {len(rows)}   median {statistics.median(r['sec'] for r in rows):.1f} s/step")
        if "peak_mem_gb" in rows[-1]:
            print(f"peak GPU memory (GB per device): {rows[-1]['peak_mem_gb']}")
        print("                         first 10%   last 10%")
        for key, label in (("forward_chrf", "forward chrF (vs ref)"), ("backward_reward", "backward reward (RL)"),
                           ("backward_kl", "KL to reference")):
            print(f"  {label:<22} {window_mean(rows, key, .1, False):9.4f}  {window_mean(rows, key, .1, True):9.4f}")
        print("  (forward chrF is noisy: temperature-1.8 samples; compare 10% windows, not single steps)")
    if bootstrap and after is not None:
        out = paired_bootstrap(run)
        if out:
            print("paired bootstrap (1st row = before, 2nd = after; '*' = p<0.05):\n" + out)


def table(root):
    print("| stage | model | lang | run | before | after | delta | paper Base -> Ours |")
    print("|---|---|---|---|---|---|---|---|")
    for m in sorted(Path(root).rglob("metrics.jsonl")):
        run = m.parent
        stage, model, lang = parse_run_dir(run)
        b, a, n = test_scores(run)
        paper = PAPER.get(model, {}).get(lang)
        ref = f"{paper[0]:.2f} -> {paper[4]:.2f}" if paper else "-"
        f = lambda x: "-" if x is None else f"{x:.2f}"
        d = "-" if (a is None or b is None) else f"{a - b:+.2f}"
        print(f"| {stage} | {model} | {lang} | {run.name} | {f(b)} | {f(a)} | {d} | {ref} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--table", metavar="RESULTS_DIR")
    ap.add_argument("--no-bootstrap", action="store_true")
    a = ap.parse_args()
    if a.table:
        table(a.table)
    for r in a.runs:
        report(r, bootstrap=not a.no_bootstrap)
