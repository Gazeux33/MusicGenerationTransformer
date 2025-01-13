from torch import nn
import torch


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.ffn_1 = nn.Linear(embed_dim, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, embed_dim)

        nn.init.xavier_uniform_(self.ffn_1.weight, gain=0.02)
        nn.init.xavier_uniform_(self.ffn_2.weight, gain=0.02)

        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.ln_1(inputs)

        attn_output, attention_scores = self.attn(
            x, x, x,
            need_weights=True,
            attn_mask=self.causal_attention_mask(inputs.size(1), inputs.size(1)).to(inputs.device)
        )

        x = inputs + self.dropout_1(attn_output)

        ff_output = self.ln_2(x)
        ff_output = self.ffn_1(ff_output)
        ff_output = torch.nn.functional.gelu(ff_output)  # GELU au lieu de ReLU
        ff_output = self.dropout_2(ff_output)
        ff_output = self.ffn_2(ff_output)

        output = x + ff_output

        return output, attention_scores

    @staticmethod
    def causal_attention_mask(n_dest, n_src):
        mask = (torch.triu(torch.ones((n_dest, n_src)), diagonal=1) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask
