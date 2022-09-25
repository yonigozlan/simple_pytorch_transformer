import os

from .inference import run_model_example, translate_example_sentence
from .training import train_model
from .vocabulary import load_tokenizers, load_vocab

if __name__ == "__main__":
    config = {
        "batch_size": 8,
        "distributed": False,
        "num_epochs": 3,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "translation_type": "de_en",
        "file_prefix": "multi30k_model_",
    }
    spacy_de, spacy_en = load_tokenizers()
    vocab_en, vocab_de = load_vocab(spacy_en, spacy_de)

    if config["translation_type"] == "en_de":
        vocab_src = vocab_en
        vocab_tgt = vocab_de
        tokenizer_src = spacy_en
        tokenizer_tgt = spacy_de
    else:
        vocab_src = vocab_de
        vocab_tgt = vocab_en
        tokenizer_src = spacy_de
        tokenizer_tgt = spacy_en

    # train_model(
    #     vocab_src,
    #     vocab_tgt,
    #     tokenizer_src,
    #     tokenizer_tgt,
    #     config,
    # )
    run_model_example(
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
        translation_type=config["translation_type"],
    )
    # translate_example_sentence(
    #     "",
    #     vocab_src,
    #     vocab_tgt,
    #     tokenizer_src,
    #     translation_type=config["translation_type"],
    # )
