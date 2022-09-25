from typing import Optional

import torch
from spacy.language import Language
from torch import Tensor, nn
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from transformer import make_transformer
from transformer.modules import EncoderDecoder
from transformer.utils import subsequent_mask

from .dataloading import Batch, create_dataloaders
from .vocabulary import tokenize_sentence


def greedy_decode(
    model: EncoderDecoder,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol_index: int,
):
    input_self_attention = model.encode(src, src_mask)
    previous_words = torch.zeros(1, 1).fill_(start_symbol_index).type_as(src.data)
    for _ in range(max_len - 1):
        output_attention = model.decode(
            input_self_attention,
            src_mask,
            previous_words,
            subsequent_mask(previous_words.size(1)).type_as(src.data),
        )
        output_probabilities = model.generator(output_attention[:, -1])
        _, next_word = torch.max(output_probabilities, dim=1)
        next_word = next_word.data[0]
        previous_words = torch.cat(
            [previous_words, torch.zeros(1, 1).type_as(src.data).fill_(next_word)],
            dim=1,
        )

    return previous_words


def infer_translation_from_tokenized_sentence(
    model: EncoderDecoder,
    input_tokenized_sentence: Tensor,
    target_tokenized_sentence: Optional[Tensor],
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    pad_index: int = 2,
    eos_string: str = "</s>",
):
    rb = Batch(input_tokenized_sentence, target_tokenized_sentence, pad_index)

    src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_index]
    tgt_tokens = (
        [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_index]
        if target_tokenized_sentence is not None
        else None
    )

    prediction_indexes = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
    predicted_text = (
        " ".join(
            [vocab_tgt.get_itos()[x] for x in prediction_indexes if x != pad_index]
        ).split(eos_string, 1)[0]
        + eos_string
    )

    return predicted_text, src_tokens, tgt_tokens


def check_outputs(
    valid_dataloader: DataLoader,
    model: nn.Module,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    n_examples: int = 15,
    pad_index: int = 2,
    eos_string: str = "</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        tokenized_example = next(iter(valid_dataloader))
        (
            predicted_text,
            src_tokens,
            tgt_tokens,
        ) = infer_translation_from_tokenized_sentence(
            model,
            tokenized_example[0],
            tokenized_example[1],
            vocab_src,
            vocab_tgt,
            pad_index,
            eos_string=eos_string,
        )
        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))
        print("Model Output               : " + predicted_text.replace("\n", ""))
        results[idx] = (src_tokens, tgt_tokens, predicted_text)

    return results


def run_model_example(
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    tokenizer_src: Language,
    tokenizer_tgt: Language,
    n_examples: int = 5,
    translation_type: str = "en_de",
):
    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        tokenizer_src,
        tokenizer_tgt,
        translation_type=translation_type,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")
    model = make_transformer(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(
            f"checkpoints/{translation_type}/multi30k_model_final.pt",
            map_location=torch.device("cpu"),
        )
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


def translate_example_sentence(
    sentence: str,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    tokenizer_src: Language,
    translation_type: str = "en_de",
):
    model = make_transformer(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(
            f"checkpoints/{translation_type}/multi30k_model_final.pt",
            map_location=torch.device("cpu"),
        )
    )
    tokenized_padded_sentence = tokenize_sentence(
        sentence, vocab_src, tokenizer_src, torch.device("cpu")
    ).unsqueeze(0)
    (predicted_text, _, _,) = infer_translation_from_tokenized_sentence(
        model, tokenized_padded_sentence, None, vocab_src, vocab_tgt
    )

    print("Model Output: " + predicted_text.replace("\n", ""))
