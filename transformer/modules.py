import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.functional import log_softmax, pad
from torchtext.vocab import Vocab

from .utils import attention, clones, subsequent_mask


class Generator(nn.Module):
    def __init__(self, model_dimension: int, vocab: Vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(model_dimension, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, nb_head: int, model_dimension: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert model_dimension % nb_head == 0
        self.d_k = model_dimension // nb_head
        self.nb_head = nb_head
        self.linears = clones(nn.Linear(model_dimension, model_dimension), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.nb_head, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attention = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.nb_head * self.d_k)

        del query
        del key
        del value

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self, model_dimension: int, feed_forward_dimension: int, dropout: float = 0.1
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.w_2 = nn.Linear(feed_forward_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.sublayer_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask)
        )

        return self.sublayer_connections[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attention: MultiHeadedAttention,
        src_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        m = memory
        x = self.sublayer_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        x = self.sublayer_connections[1](
            x, lambda x: self.src_attention(x, m, m, src_mask)
        )

        return self.sublayer_connections[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.layer_norm = LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.layer_norm(x)


class Embeddings(nn.Module):
    def __init__(self, model_dimension: int, vocab: Vocab):
        super(Embeddings, self).__init__()
        self.lookup_table = nn.Embedding(vocab, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, x: Tensor) -> Tensor:
        return self.lookup_table(x) * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(max_len, model_dimension)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2) * -(math.log(10000.0) / model_dimension)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.positional_encoding[:, : x.size(1)].requires_grad_(False)

        return self.dropout(x)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: Tensor,
        src_mask: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
    ):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
