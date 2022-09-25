import os
from os.path import exists
from typing import Any, Callable, Iterable

import spacy
import torch
import torchtext.datasets as datasets
from spacy.language import Language
from torch.nn.functional import pad
from torchtext.vocab import Vocab, build_vocab_from_iterator

# Load spacy tokenizer models, download them if they haven't been
# downloaded already


def load_tokenizers() -> tuple[Language, Language]:
    try:
        tokenizer_src = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        tokenizer_src = spacy.load("de_core_news_sm")

    try:
        tokenizer_tgt = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        tokenizer_tgt = spacy.load("en_core_web_sm")

    return tokenizer_src, tokenizer_tgt


def tokenize(text: str, tokenizer: Language):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def tokenize_sentence(
    sentence: str,
    vocab: Vocab,
    tokenizer: Language,
    device: str,
    max_padding: int = 128,
    pad_index: int = 2,
):
    beginning_sentence_index = torch.tensor([0], device=device)  # <s> token id
    end_of_sentence_index = torch.tensor([1], device=device)  # </s> token id

    tokenized_sentence = torch.cat(
        [
            beginning_sentence_index,
            torch.tensor(
                vocab(tokenize(sentence, tokenizer)),
                dtype=torch.int64,
                device=device,
            ),
            end_of_sentence_index,
        ],
        0,
    )

    return pad(
        tokenized_sentence,
        (
            0,
            max_padding - len(tokenized_sentence),
        ),
        value=pad_index,
    )


def yield_tokens(data_iter: Iterable, tokenizer: Callable, index: int):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(tokenizer_en: Language, tokenizer_de: Language):
    def tokenize_en(text: str):
        return tokenize(text, tokenizer_en)

    def tokenize_de(text: str):
        return tokenize(text, tokenizer_de)

    train, val, test = datasets.Multi30k(language_pair=("en", "de"))
    print("Building English vocabulary ...")
    vocab_en = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building German vocabulary ...")
    vocab_de = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_en.set_default_index(vocab_en["<unk>"])
    vocab_de.set_default_index(vocab_de["<unk>"])

    return vocab_en, vocab_de


def load_vocab(tokenizer_en: Language, tokenizer_de: Language):
    if not exists("data/vocab_en_de.pt"):
        os.mkdir("data")
        vocab_en, vocab_de = build_vocabulary(tokenizer_en, tokenizer_de)
        torch.save((vocab_en, vocab_de), "data/vocab_en_de.pt")
    else:
        vocab_en, vocab_de = torch.load("data/vocab_en_de.pt")

    return vocab_en, vocab_de
