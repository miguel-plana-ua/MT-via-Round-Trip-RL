import inspect
import os
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset, Sampler
import warnings
from datasets import load_dataset, load_from_disk

# --- Compatibility shims (replication) -------------------------------------------------------
# The original code imports torchdata.datapipes (removed in torchdata>=0.10) and pytorch_lightning.
# If they are installed they are used exactly as before; otherwise we fall back to equivalent
# plain-PyTorch stand-ins (a MapDataPipe is just a map-style Dataset; the data module only uses
# LightningDataModule.save_hyperparameters).
try:
    from torchdata.datapipes.map import MapDataPipe
except ImportError:
    MapDataPipe = Dataset

try:
    from pytorch_lightning import LightningDataModule
except ImportError:
    class LightningDataModule:
        def save_hyperparameters(self, ignore=None):
            caller_locals = inspect.currentframe().f_back.f_locals
            skip = {"self", "__class__", ignore}
            self.hparams = SimpleNamespace(**{k: v for k, v in caller_locals.items() if k not in skip})

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class TranslationDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        illegal_token_mask,
        data_path,
        dataset_config_name,
        source_lang,
        target_lang,
        sort_by_length: bool = True,
        sort_direction: str = "asc",
        train_batch_size: int = 1,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        subset_seed: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self._train_sampler = None
        self._train_batch_size = max(int(train_batch_size), 1)

    def setup(self, stage=None):
        # [replication] The HF loading script downloads the data from tinyurl.com links on every run
        # (even offline), which is fragile on a cluster. If `data_path` is a local directory holding a copy
        # saved with scripts/prefetch.py (<data_path>/<config_name>/), that copy is used instead.
        local_copy = os.path.join(str(self.hparams.data_path), self.hparams.dataset_config_name)
        if os.path.isdir(local_copy):
            prompts = load_from_disk(local_copy)
        else:
            prompts = load_dataset(
                self.hparams.data_path, self.hparams.dataset_config_name, trust_remote_code=True
            )

        def _resolve_split(split_names, limit=None):
            for split_name in split_names:
                if split_name in prompts:
                    split = prompts[split_name]
                    if limit is not None and int(limit) < len(split):
                        # Seeded random subset (not the first N rows: the data mixes 3 domains, a prefix could be biased).
                        split = split.shuffle(seed=self.hparams.subset_seed).select(range(int(limit)))
                    return split
            return None

        train_split = _resolve_split(("train", "training"), self.hparams.max_train_samples)
        if train_split is None:
            raise ValueError("Training split not found in dataset.")
        self.train_data = TranslationDataPipe(
            train_split,
            self.tokenizer,
            src_col="sentence_" + self.hparams.source_lang,
            tgt_col="sentence_" + self.hparams.target_lang,
        )

        val_split = _resolve_split(("valid", "validation", "val"), self.hparams.max_val_samples)
        if val_split is None:
            raise ValueError("Validation split not found in dataset.")
        self.val_data = TranslationDataPipe(
            val_split,
            self.tokenizer,
            src_col="sentence_" + self.hparams.source_lang,
            tgt_col="sentence_" + self.hparams.target_lang,
        )

        test_split = _resolve_split(("test", "test_final"), self.hparams.max_test_samples)
        if test_split is not None:
            self.test_data = TranslationDataPipe(
                test_split,
                self.tokenizer,
                src_col="sentence_" + self.hparams.source_lang,
                tgt_col="sentence_" + self.hparams.target_lang,
            )
        else:
            self.test_data = None

    def train_dataloader(self):
        if self._train_sampler is not None:
            return DataLoader(
                self.train_data,
                sampler=self._train_sampler,
                batch_size=self._train_batch_size,
                num_workers=0,
                collate_fn=self._collate_batch,
            )
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self._train_batch_size,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=48,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self):
        if self.test_data is None:
            return None
        return DataLoader(
            self.test_data,
            batch_size=48,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch):
        encoder_texts, targets, sources, sample_ids = zip(*batch)
        batch_encoding = self.tokenizer(
            list(encoder_texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return batch_encoding, list(targets), list(sources), list(sample_ids)


class TranslationDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer, src_col, tgt_col) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.src_col = src_col
        self.tgt_col = tgt_col

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        src_text = self.prompts[index][self.src_col]
        tgt_text = self.prompts[index][self.tgt_col]
        return src_text, tgt_text, src_text, index
