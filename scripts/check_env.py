"""Environment sanity check: python -- scripts/check_env.py [--load-model]

Prints library versions and what each visible GPU is (name, total memory) and checks that a CUDA kernel runs.
GPU memory decides which --mem profile of scripts/run_stage.py to use (see docs/replicacion/03_SERVIDOR.md).
"""
import argparse
import importlib
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-model", action="store_true", help="also load tokenizer+model from the HF cache")
    args = ap.parse_args()

    print("python", sys.version.split()[0])
    for name in ("torch", "transformers", "datasets", "sacrebleu", "hydra", "omegaconf", "sentencepiece",
                 "bitsandbytes", "wandb"):
        try:
            mod = importlib.import_module(name)
            print(f"{name:14s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:14s} NOT AVAILABLE ({type(exc).__name__})")

    import torch

    print("torch.version.cuda", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("visible GPUs:", n)
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {prop.name}  {prop.total_memory / 2**30:.1f} GiB  sm_{prop.major}{prop.minor}")
        x = torch.randn(1024, 1024, device=f"cuda:{i}")
        torch.cuda.synchronize(i)
        print(f"    matmul ok ({float((x @ x).sum()) != 0})")
    if n == 1:
        print("NOTE: with 1 GPU, policy and reference model share it (main.py uses cuda:1 / cuda:0 when n>=2).")
    if n == 0:
        print("NOTE: no GPU visible.")

    if args.load_model:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        enc = tok("The weather is nice today.", return_tensors="pt")
        out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("rus_Cyrl"), max_new_tokens=32)
        print("sample translation:", tok.batch_decode(out, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
