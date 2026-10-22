"""Download everything the experiments need (run on a node WITH internet, e.g. the login node).

    python scripts/prefetch.py --models 600M --langs ayr_Latn dyu_Latn wol_Latn rus_Cyrl

* model + tokenizer go to the HF cache (HF_HOME);
* each dataset config is saved to `data/nllb-md/<config>/` with `save_to_disk`. scripts/run_stage.py then uses that
  local copy automatically (see dl.py): no network and no tinyurl.com links are needed afterwards. The folder is
  plain files, so it can also be copied between machines with scp/rsync.
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"600M": "facebook/nllb-200-distilled-600M", "1.3B": "facebook/nllb-200-distilled-1.3B"}
LANGS = ["ayr_Latn", "fur_Latn", "wol_Latn", "bho_Deva", "dyu_Latn", "rus_Cyrl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["600M"], choices=list(MODELS), help="empty list = skip models")
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "nllb-md"))
    args = ap.parse_args()

    for key in args.models:
        print("model", MODELS[key])
        AutoTokenizer.from_pretrained(MODELS[key])
        AutoModelForSeq2SeqLM.from_pretrained(MODELS[key])
    for lang in args.langs:
        config = f"eng_Latn-{lang}"
        out = Path(args.data_dir) / config
        if out.is_dir():
            print(f"dataset {config}: already saved in {out}")
            continue
        print("dataset", config)
        ds = load_dataset("breakend/nllb-multi-domain", config, trust_remote_code=True)
        print("  ", {k: len(v) for k, v in ds.items()}, ds["train"].column_names)
        ds.save_to_disk(str(out))
    print("prefetch done")


if __name__ == "__main__":
    main()
