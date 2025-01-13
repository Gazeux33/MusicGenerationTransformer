from torch import nn
import torch

from src.transformer_block import TransformerBlock
from src.embedding import TokenAndPositionEmbedding


class MusicModel(nn.Module):
    def __init__(self, notes_vocab_size,
                 durations_vocab_size,
                 embed_dim,
                 num_heads,
                 key_dim,
                 feed_forward_dim,
                 dropout_rate):
        super(MusicModel, self).__init__()

        self.notes_embedding = TokenAndPositionEmbedding(
            vocab_size=notes_vocab_size,
            embed_dim=embed_dim // 2
        )

        self.durations_embedding = TokenAndPositionEmbedding(
            vocab_size=durations_vocab_size,
            embed_dim=embed_dim // 2
        )

        self.block = TransformerBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            embed_dim=embed_dim,
            ff_dim=feed_forward_dim,
            dropout_rate=dropout_rate
        )

        self.out_note = nn.Linear(embed_dim, notes_vocab_size)
        self.out_duration = nn.Linear(embed_dim, durations_vocab_size)

        nn.init.xavier_uniform_(self.out_note.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_duration.weight, gain=0.02)

    def forward(self, note, duration):
        note_embedding = self.notes_embedding(note)
        duration_embedding = self.durations_embedding(duration)

        x = torch.cat([note_embedding, duration_embedding], dim=-1)
        x, attention_score = self.block(x)

        out_note = torch.log_softmax(self.out_note(x), dim=-1)
        out_duration = torch.log_softmax(self.out_duration(x), dim=-1)

        return out_note, out_duration, attention_score
