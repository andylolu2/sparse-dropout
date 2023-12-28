import string
from pathlib import Path

import lightning as L
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from torch.utils.data import DataLoader


class WikitextDataModule(L.LightningDataModule):
    dataset: Dataset
    train_dataset: Dataset
    val_dataset: Dataset
    tokenzier: Tokenizer

    def __init__(
        self,
        train_batch_size: int,
        length: int,
        cache_dir: Path,
        vocab_size: int,
        batch_first: bool,
        train_size: int = 5000,
        val_size: int = 1000,
        val_batch_size: int | None = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.length = length
        self.vocab_size = vocab_size
        self.batch_first = batch_first
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size or train_batch_size
        self.train_size = train_size
        self.val_size = val_size

    @property
    def train_samlpe(self):
        return self.train_dataset[0]

    def prepare_data(self):
        dataset = load_dataset(
            "EleutherAI/wikitext_document_level",
            "wikitext-103-raw-v1",
            split="train",
            cache_dir=str(self.cache_dir),
        )
        assert isinstance(dataset, Dataset)

        tokenizer_path = self.cache_dir / "tokenizer.json"

        if tokenizer_path.exists():
            print(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = Tokenizer(models.BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )

            def batch_iterator(batch_size: int):
                for i in range(0, len(dataset), batch_size):
                    yield dataset[i : i + batch_size]["page"]

            tokenizer.train_from_iterator(
                batch_iterator(batch_size=1000),
                trainer=trainer,
                length=len(dataset),
            )

            tokenizer.save(str(self.cache_dir / "tokenizer.json"))
            self.tokenizer = tokenizer

        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

        def tokenize_and_chunk(batch):
            tokens_batch = self.tokenizer.encode_batch(batch["page"])

            chunks = []
            for encoding in tokens_batch:
                tokens = encoding.ids
                for i in range(0, len(tokens) - self.length, self.length):
                    chunks.append(tokens[i : i + self.length])

            return {"tokens": chunks}

        self.dataset = dataset.map(
            tokenize_and_chunk,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names,
            num_proc=8,
        )

    def setup(self, stage: str):
        if stage == "fit":
            indices = np.random.choice(
                len(self.dataset), self.train_size + self.val_size, replace=False
            ).tolist()
            self.train_dataset = self.dataset.select(indices[: self.train_size])
            self.val_dataset = self.dataset.select(indices[self.train_size :])
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def collate_fn(self, batch: list[dict]):
        tokens_batch = torch.tensor([b["tokens"] for b in batch], dtype=torch.long)
        if self.batch_first:
            return tokens_batch
        else:
            return tokens_batch.transpose(0, 1).contiguous()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
        )


if __name__ == "__main__":
    L.seed_everything(0)

    dm = WikitextDataModule(
        train_batch_size=4,
        length=128,
        cache_dir=Path("data", "wikitext"),
        batch_first=True,
    )
    dm.prepare_data()

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    for batch in train_loader:
        print(batch.shape)
        for text in dm.tokenizer.decode_batch(batch.tolist()):
            print(text)
        break
