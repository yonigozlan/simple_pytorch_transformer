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
    N=6,
    d_model=512,
    feed_forward_dimension=2048,
    nb_head=8,
    dropout=0.1,
):
    attention_layer = MultiHeadedAttention(nb_head, d_model)
    feed_forward_layer = PositionwiseFeedForward(
        d_model, feed_forward_dimension, dropout
    )
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                d_model,
                deepcopy(attention_layer),
                deepcopy(feed_forward_layer),
                dropout,
            ),
            N,
        ),
        Decoder(
            DecoderLayer(
                d_model,
                deepcopy(attention_layer),
                deepcopy(attention_layer),
                deepcopy(feed_forward_layer),
                dropout,
            ),
            N,
        ),
        nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model
