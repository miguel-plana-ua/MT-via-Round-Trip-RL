"""Launch one replication stage of `main.py` (the paper's method, "Ours") with sane presets.

    python scripts/run_stage.py smoke --lang ayr_Latn --mem lowmem
    python scripts/run_stage.py base  --lang ayr_Latn            # Base chrF++ only, no training
    python scripts/run_stage.py pilot --lang ayr_Latn --seed 27
    python scripts/run_stage.py full  --lang ayr_Latn

Stages (see docs/replicacion/02_PLAN.md):
  smoke : tiny run (40 train sentences, 20 steps: crosses the step-16 reference refresh). Validates the pipeline, memory and speed. No learning expected.
  base  : evaluation only (val + test, full splits) of the untouched NLLB model -> "Base" column of Table 1.
  pilot : 600 optimisation steps (1200 sentences), full test before/after. First test of "does it improve?".
  full  : paper setting: 2 epochs over the 6000 training sentences, full test before/after.

Output: results/<stage>/<model>/<lang>/<tag>seed<seed>/  (log.txt, metrics.jsonl, train_log.jsonl, hyps/).
A DONE marker is written when a run finishes; finished runs are skipped unless --force.
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

# Hydra overrides per stage (everything else = configs/task/nllb_md.yaml = the paper's hyper-parameters)
STAGES = {
    "smoke": [
        "task.data.max_train_samples=40", "task.data.max_val_samples=32", "task.data.max_test_samples=32",
        "task.training.epochs=1", "task.training.max_steps=20", "task.eval.every_n_opt_steps=10",
        "task.training.save_model=false",
    ],
    "base": [
        "task.eval.only=true", "task.training.save_model=false",
    ],
    "pilot": [
        "task.training.epochs=1", "task.training.max_steps=600",
        "task.data.max_val_samples=200", "task.eval.every_n_opt_steps=100",
        "task.training.save_model=false",
    ],
    "full": [
        "task.training.epochs=2",
        "task.data.max_val_samples=300", "task.eval.every_n_opt_steps=500",
        "task.training.save_model=true",
    ],
}

# GPU-memory profiles. "paper" changes nothing. The others are OPT-IN deviations, see docs.
MEM = {
    "paper": [],
    "lowmem": ["task.training.freeze_embeddings=true", "task.training.logp_chunk_groups=2"],
    "lowmem1": ["task.training.freeze_embeddings=true", "task.training.logp_chunk_groups=1"],
    "tight": ["task.training.freeze_embeddings=true", "task.training.logp_chunk_groups=1",
              "task.training.optimizer=adamw8bit"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--lang", default="ayr_Latn", help="target language (NLLB code), source is always eng_Latn")
    p.add_argument("--model", default="600M", choices=list(MODELS))
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--mem", default="lowmem", choices=list(MEM))
    p.add_argument("--lr", type=float, default=None, help="override learning rate (paper: 2e-6)")
    p.add_argument("--max-steps", type=int, default=None, help="override the stage's max_steps")
    p.add_argument("--tag", default="", help="free label added to the output directory name")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra raw Hydra overrides")
    p.add_argument("--offline", action="store_true", help="HF_HUB_OFFLINE=1 (use the pre-fetched cache)")
    p.add_argument("--force", action="store_true", help="re-run even if DONE exists")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    name = f"{args.tag}{'_' if args.tag else ''}seed{args.seed}"
    out_dir = Path(args.results_dir).resolve() / args.stage / args.model / args.lang / name
    overrides = [
        f"hydra.run.dir={out_dir.as_posix()}",
        f"seed={args.seed}",
        f"task.model.name={MODELS[args.model]}",
        f"task.data.target_lang={args.lang}",
        f"task.data.dataset_config_name=eng_Latn-{args.lang}",
        "task.training.use_wandb=false",
        *STAGES[args.stage],
        *(MEM[args.mem] if args.stage != "base" else []),
    ]
    local_data = ROOT / "data" / "nllb-md"  # created by scripts/prefetch.py; avoids the tinyurl download
    if (local_data / f"eng_Latn-{args.lang}").is_dir():
        overrides.append(f"task.data.path={local_data.as_posix()}")
    if args.lr is not None:
        overrides.append(f"task.training.lr={args.lr}")
    if args.max_steps is not None:
        overrides.append(f"task.training.max_steps={args.max_steps}")
    overrides += args.set

    cmd = [sys.executable, str(ROOT / "main.py"), *overrides]
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    if (out_dir / "DONE").exists() and not args.force:
        print(f"{out_dir} already DONE, skipping (use --force)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("DONE", "metrics.jsonl", "train_log.jsonl"):  # main.py appends: never mix two runs
        (out_dir / stale).unlink(missing_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    (out_dir / "cmd.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    with open(out_dir / "log.txt", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"FAILED (exit {rc}); see {out_dir / 'log.txt'}")
        return rc
    (out_dir / "DONE").write_text("ok\n")
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
