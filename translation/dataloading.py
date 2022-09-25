from typing import Optional

import torch
import torchtext.datasets as datasets
from spacy.language import Language
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import Vocab

from transformer.utils import make_std_mask

from .vocabulary import tokenize_sentence


class Batch:
    def __init__(self, src: Tensor, tgt: Optional[Tensor] = None, pad_index: int = 2):
        self.src = src
        self.src_mask = (src != pad_index).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = make_std_mask(self.tgt, pad_index)
            self.ntokens = (self.tgt_y != pad_index).data.sum()


def collate_batch(
    batch: list[str],
    tokenizer_src: Language,
    tokenizer_tgt: Language,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str,
    max_padding: int = 128,
    pad_index: int = 2,
):
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_list.append(
            tokenize_sentence(
                _src,
                src_vocab,
                tokenizer_src,
                device,
                max_padding=max_padding,
                pad_index=pad_index,
            )
        )
        tgt_list.append(
            tokenize_sentence(
                _tgt,
                tgt_vocab,
                tokenizer_tgt,
                device,
                max_padding=max_padding,
                pad_index=pad_index,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    return (src, tgt)


def create_dataloaders(
    device: torch.device,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    tokenizer_src: Language,
    tokenizer_tgt: Language,
    translation_type: str = "en_de",
    batch_size: int = 1000,
    max_padding: int = 128,
    is_distributed: bool = True,
):
    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenizer_src,
            tokenizer_tgt,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_index=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=tuple(translation_type.split("_"))
    )

    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader
