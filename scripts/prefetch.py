"""Download everything the experiments need (run on a node WITH internet, e.g. the login node).

    python scripts/prefetch.py --models 600M --langs ayr_Latn dyu_Latn wol_Latn rus_Cyrl

* model + tokenizer go to the HF cache (HF_HOME);
* each dataset config is saved to `data/nllb-md/<config>/` with `save_to_disk`. scripts/run_stage.py then uses that
  local copy automatically (see dl.py): no network and no tinyurl.com links are needed afterwards. The folder is
  plain files, so it can also be copied between machines with scp/rsync.

Dataset source (--source):
  script  : the official loading script of breakend/nllb-multi-domain; it downloads from tinyurl.com links, which
            did NOT respond from our machines (checked 2026-10-22, both from a PC and from the server).
  parquet : the Parquet export that Hugging Face keeps in the branch `refs/convert/parquet` of the same dataset
            (same rows, same order, same columns). Needs only huggingface.co.
  auto    : try `script`, fall back to `parquet` (default).
In every case the sizes (6000 / 1309 / 1500) and the column names are printed so you can check them.
"""
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
REPO = "breakend/nllb-multi-domain"
MODELS = {"600M": "facebook/nllb-200-distilled-600M", "1.3B": "facebook/nllb-200-distilled-1.3B"}
LANGS = ["ayr_Latn", "fur_Latn", "wol_Latn", "bho_Deva", "dyu_Latn", "rus_Cyrl"]


def from_script(config):
    return load_dataset(REPO, config, trust_remote_code=True)


def from_parquet(config):
    from huggingface_hub import hf_hub_download, list_repo_files

    rev = "refs/convert/parquet"
    files = list_repo_files(REPO, repo_type="dataset", revision=rev)
    splits = {}
    for split in ("train", "valid", "test"):
        parts = sorted(f for f in files if f.startswith(f"{config}/{split}/") and f.endswith(".parquet"))
        if not parts:
            raise RuntimeError(f"no parquet files for {config}/{split} in {rev}; files seen: {files[:10]}")
        paths = [hf_hub_download(REPO, f, repo_type="dataset", revision=rev) for f in parts]
        splits[split] = Dataset.from_parquet(paths)
    return DatasetDict(splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["600M"], choices=list(MODELS), help="empty list = skip models")
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "nllb-md"))
    ap.add_argument("--source", default="auto", choices=["auto", "script", "parquet"])
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
        if args.source == "script":
            ds = from_script(config)
        elif args.source == "parquet":
            ds = from_parquet(config)
        else:
            try:
                ds = from_script(config)
            except Exception as exc:  # noqa: BLE001
                print(f"  loading script failed ({type(exc).__name__}: {str(exc)[:100]}); using the parquet export")
                ds = from_parquet(config)
        print("  sizes:", {k: len(v) for k, v in ds.items()}, "| columns:", ds["train"].column_names)
        print("  domains at rows 0/3000/5999 of train:", [ds["train"][i]["domain"] for i in (0, 3000, 5999)])
        ds.save_to_disk(str(out))
    print("prefetch done")


if __name__ == "__main__":
    main()
