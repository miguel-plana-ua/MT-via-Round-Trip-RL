"""Run every experiment of the paper: {ours, bt, rt, umt} x {600M, 1.3B} x 6 languages.

Usage (from the repo root):
    uv run python scripts/run_all.py --dry-run
    uv run python scripts/run_all.py --smoke --models 600M --langs rus_Cyrl --methods ours bt
    uv run python scripts/run_all.py --models 600M 1.3B
Results go to results/<method>/<model>/<lang>/ (log.txt, plus metrics.jsonl for "ours").
Finished runs (marker file DONE) are skipped, so the script can be re-launched after a crash.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "600M": "facebook/nllb-200-distilled-600M",
    "1.3B": "facebook/nllb-200-distilled-1.3B",
}
LANGS = ["ayr_Latn", "fur_Latn", "wol_Latn", "bho_Deva", "dyu_Latn", "rus_Cyrl"]
BASELINE_SCRIPTS = {
    "bt": ROOT / "baselines" / "backtranslation" / "backtranslation.py",
    "rt": ROOT / "baselines" / "roundtrip-bt" / "roundtrip-bt.py",
    "umt": ROOT / "baselines" / "UMNMT" / "umnmt.py",
}


def build_command(method, model_key, lang, out_dir, smoke, wandb):
    model = MODELS[model_key]
    pair = f"eng_Latn-{lang}"
    if method == "ours":
        cmd = [sys.executable, str(ROOT / "main.py"),
               f"hydra.run.dir={out_dir}",
               f"task.model.name={model}",
               f"task.data.target_lang={lang}",
               f"task.data.dataset_config_name={pair}",
               f"task.training.use_wandb={str(wandb and not smoke).lower()}"]
        if smoke:
            cmd += ["task.data.max_train_samples=4", "task.data.max_eval_samples=4",
                    "task.training.epochs=1", "task.training.save_model=false"]
        return cmd

    num_epochs = "1" if smoke else "2"
    cmd = [sys.executable, str(BASELINE_SCRIPTS[method]),
           "--dataset_config", pair, "--src_lang_nllb", "eng_Latn",
           "--tgt_lang_nllb", lang, "--model_name", model,
           "--num_epochs", num_epochs, "--batch_size", "2"]
    if method in ("bt", "rt"):
        cmd += ["--bt_batch_size", "2", "--learning_rate", "3e-5", "--regenerate_bt_every", "1"]
    if smoke:
        cmd += ["--max_samples", "4", "--max_val_samples", "4", "--max_test_samples", "4"]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--langs", nargs="+", default=LANGS)
    p.add_argument("--methods", nargs="+", default=["ours", "bt", "rt", "umt"],
                   choices=["ours", "bt", "rt", "umt"])
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for model_key in args.models:
        for lang in args.langs:
            for method in args.methods:
                name = ("smoke_" + method) if args.smoke else method
                out_dir = Path(args.results_dir) / name / model_key / lang
                cmd = build_command(method, model_key, lang, out_dir, args.smoke, not args.no_wandb)
                print(f"[{name}/{model_key}/{lang}] {' '.join(cmd)}")
                if args.dry_run:
                    continue
                if (out_dir / "DONE").exists():
                    print("  already done, skipping")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "log.txt", "w", encoding="utf-8") as log:
                    rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"  FAILED (exit {rc}); see {out_dir / 'log.txt'}")
                    continue
                (out_dir / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
