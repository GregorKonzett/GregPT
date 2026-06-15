from torch import nn
import torch
from torch.functional import F

from tokenizer.TikTokenTokenizer import TikTokenTokenizer

dropout = 0.02
dropout1 = 0.02
embed_dim = 512
num_heads = 8
head_size = embed_dim // num_heads
block_count = 8
block_size = 256

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ks = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.qs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def apply_rope(self, x):
        B, H, T, hs = x.shape

        pos = torch.arange(T, device=x.device)
        dim = torch.arange(0, hs, 2, device=x.device)

        inv_freq = 1.0 / (10000 ** (dim.float() / hs))
        freqs = torch.outer(pos.float(), inv_freq)  # (T, hs/2)

        cos = freqs.cos()[None, None, :, :]  # (1, 1, T, hs/2)
        sin = freqs.sin()[None, None, :, :]  # (1, 1, T, hs/2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.stack(
            [x_even * cos - x_odd * sin,
             x_even * sin + x_odd * cos],
            dim=-1
        )

        return x_rotated.flatten(-2)

    def forward(self, x):
        B, T, C = x.shape

        ks = self.ks(x).view(B, T, num_heads, head_size).transpose(1, 2)
        vs = self.vs(x).view(B, T, num_heads, head_size).transpose(1, 2)
        qs = self.qs(x).view(B, T, num_heads, head_size).transpose(1, 2)

        qs = self.apply_rope(qs)
        ks = self.apply_rope(ks)

        wei = qs @ ks.transpose(-2, -1) * ks.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ vs
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.dropout1(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, idx):
        return self.net(idx)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_norm1 = nn.LayerNorm(embed_dim)
        self.ln_norm2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward()
        self.multihead_attn = MultiHeadAttention()

    def forward(self, x):
        x = x + self.multihead_attn(self.ln_norm1(x))
        x = x + self.ffwd(self.ln_norm2(x))
        return x

class GptModel(nn.Module):
    def __init__(self, vocab_size, tokenizer: TikTokenTokenizer):
        super().__init__()
        self.device = get_device()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.block_count = block_count
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embed_dim, device=self.device)
        self.blocks = nn.Sequential(*[Block() for _ in range(self.block_count)])
        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
        self.ln_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.token_embedding_table(x)
        x = self.blocks(x)
        x = self.ln_norm(x)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, temperature = 0.7):
        self.eval()
        for _ in range(block_size):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            next_token = idx_next[0, 0].item()
            if self.tokenizer.is_eos(next_token):
                break

            idx = torch.cat([idx, idx_next], dim=1)

        self.train()

        return idx
