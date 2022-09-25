import argparse

from .inference import run_model_example, translate_example_sentence
from .training import train_model
from .vocabulary import load_tokenizers, load_vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")

    train_parser = subparser.add_parser("train")
    examples_parser = subparser.add_parser("examples")
    infer_parser = subparser.add_parser("infer")

    train_parser.add_argument("--language", type=str, default="en_de")
    train_parser.add_argument("--epochs", type=int, default=8)

    examples_parser.add_argument("--language", type=str, default="en_de")
    examples_parser.add_argument("--nb", type=int, default=5)

    infer_parser.add_argument("sentence", type=str)
    infer_parser.add_argument("--language", type=str, default="en_de")

    args = parser.parse_args()
    spacy_de, spacy_en = load_tokenizers()
    vocab_en, vocab_de = load_vocab(spacy_en, spacy_de)
    config = {
        "batch_size": 8,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "translation_type": args.language,
        "file_prefix": "multi30k_model_",
    }

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

    if args.command == "infer":
        translate_example_sentence(
            args.sentence,
            vocab_src,
            vocab_tgt,
            tokenizer_src,
            translation_type=config["translation_type"],
        )
    elif args.command == "examples":
        run_model_example(
            vocab_src,
            vocab_tgt,
            tokenizer_src,
            tokenizer_tgt,
            n_examples=args.nb,
            translation_type=config["translation_type"],
        )
    elif args.command == "train":
        config["num_epochs"] = args.epochs
        train_model(
            vocab_src,
            vocab_tgt,
            tokenizer_src,
            tokenizer_tgt,
            config,
        )
