import math
from collections import OrderedDict

import torch as th
from torch import nn

# Stealing very extensively from: https://pytorch.org/tutorials/beginner/translation_transformer.html
# Ignoring all padding bits for now


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: th.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = th.exp(- th.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = th.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = th.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = th.sin(pos * den)
        pos_embedding[:, 1::2] = th.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: th.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, nhead, decoder_layers,
                 feedforward_dim, dropout, n_classes):
        super().__init__()
        self.token_embedder = TokenEmbedding(n_classes, embed_dim)
        self.positional_encoder = PositionalEncoding(embed_dim, dropout=dropout)

        blocks = OrderedDict()
        for i in range(decoder_layers):
            blocks[f'transformer_block_{i}'] = TransformerDecoderBlock(embed_dim, nhead, dropout, feedforward_dim)
        self.transformer_blocks = nn.Sequential(blocks)

        self.linear_layer = nn.Linear(embed_dim, n_classes)
        self.softmax = nn.Softmax(dim=2)

        self.embedding_dim = embed_dim

    def forward(self, x):
        x = self.token_embedder(x)
        x = self.positional_encoder(x)
        x = self.transformer_blocks(x)
        logits = self.linear_layer(x)
        output_probs = self.softmax(logits)

        return output_probs


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, dropout, feedforward_dim):
        super().__init__()
        self.attention_block = nn.MultiheadAttention(embed_dim=embed_dim,
                                                     num_heads=num_attention_heads,
                                                     dropout=dropout,
                                                     batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.linear_1 = nn.Linear(embed_dim, feedforward_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(feedforward_dim, embed_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        mask = (th.triu(th.ones((seq_len, seq_len), device=x.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        x = x + self.attention_block(x, x, x,
                                     attn_mask=mask,
                                     need_weights=False)[0]
        x = self.layer_norm_1(x)
        x = x + self.feedforward(x)
        x = self.layer_norm_2(x)
        return x

    def feedforward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
