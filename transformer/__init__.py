from copy import deepcopy

from torch import nn
from torchtext.vocab import Vocab

from .modules import (
    Decoder,
    DecoderLayer,
    Embeddings,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    Generator,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)


def make_transformer(
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    N: int = 6,
    model_dimension: int = 512,
    feed_forward_dimension: int = 2048,
    nb_head: int = 8,
    dropout: float = 0.1,
):
    attention_layer = MultiHeadedAttention(nb_head, model_dimension)
    feed_forward_layer = PositionwiseFeedForward(
        model_dimension, feed_forward_dimension, dropout
    )
    position = PositionalEncoding(model_dimension, dropout)
    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                model_dimension,
                deepcopy(attention_layer),
                deepcopy(feed_forward_layer),
                dropout,
            ),
            N,
        ),
        Decoder(
            DecoderLayer(
                model_dimension,
                deepcopy(attention_layer),
                deepcopy(attention_layer),
                deepcopy(feed_forward_layer),
                dropout,
            ),
            N,
        ),
        nn.Sequential(Embeddings(model_dimension, src_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(model_dimension, tgt_vocab), deepcopy(position)),
        Generator(model_dimension, tgt_vocab),
    )

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model
