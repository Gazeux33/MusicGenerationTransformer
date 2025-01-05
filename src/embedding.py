import torch
import torch.nn as nn

class SinePositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(SinePositionEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :].unsqueeze(0)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=5000):
        super(TokenAndPositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.pos_emb = SinePositionEncoding(embed_dim, max_len)

    def forward(self, x):
        # x: Tensor de forme (batch_size, seq_len)
        embedding = self.token_emb(x)
        positions = self.pos_emb(x)
        return embedding + positions
