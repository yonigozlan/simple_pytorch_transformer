import os
import time
from typing import Any

import GPUtil
import torch
import torch.distributed as dist
from spacy.language import Language
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchtext.vocab import Vocab

from transformer import make_transformer
from transformer.loss import LabelSmoothing, SimpleLossCompute
from transformer.utils import rate

from .dataloading import Batch, create_dataloaders


class DummyOptimizer(Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter: tuple[Batch],
    model: nn.Module,
    loss_compute: SimpleLossCompute,
    optimizer: Optimizer,
    scheduler: Any,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState = TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node

    return total_loss / total_tokens, train_state


def train_worker(
    gpu: int,
    ngpus_per_node: int,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    tokenizer_src: Language,
    tokenizer_tgt: Language,
    config: dict,
    is_distributed: bool = False,
    save_path="",
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_index = vocab_tgt["<blank>"]
    d_model = 512
    model = make_transformer(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model

    criterion = LabelSmoothing(size=len(vocab_tgt), pad_index=pad_index, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
        translation_type=config["translation_type"],
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_index) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        file_path = os.path.join(
            save_path, "%s%.2d.pt" % (config["file_prefix"], epoch)
        )
        torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_index) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    file_path = os.path.join(save_path, "%sfinal.pt" % config["file_prefix"])
    torch.save(module.state_dict(), file_path)


def train_model(
    vocab_src,
    vocab_tgt,
    tokenizer_src,
    tokenizer_tgt,
    config,
):
    save_path = f"checkpoints/{config['translation_type']}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_worker(
        0,
        1,
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
        config,
        False,
        save_path=save_path,
    )
