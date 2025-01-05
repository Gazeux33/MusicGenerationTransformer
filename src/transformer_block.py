from torch import nn
import torch



class TransformerBlock(nn.Module):
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.ffn_1 = nn.Linear(embed_dim, ff_dim)
        self.ffn_2 = nn.Linear(ff_dim, embed_dim)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)

    @staticmethod
    def causal_attention_mask(n_dest, n_src):
        i = torch.arange(n_dest).unsqueeze(1)
        j = torch.arange(n_src).unsqueeze(0)
        mask = (i >= j - n_src + n_dest)
        return mask

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        causal_mask = self.causal_attention_mask(seq_len, seq_len).to(inputs.device)
        attention_output, attention_scores = self.attn(inputs, inputs, inputs, attn_mask=causal_mask)
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = torch.relu(self.ffn_1(out1))
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return self.ln_2(out1 + ffn_output), attention_scores

