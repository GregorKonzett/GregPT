from torch import nn
import torch
from torch.functional import F

from tokenizer import Tokenizer

dropout = 0.02
embed_dim = 384
num_heads = 6
head_size = embed_dim // num_heads
block_count = 6
block_size = 256

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T, C = idx.shape
        k = self.key(idx)
        v = self.value(idx)
        q = self.query(idx)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        out = torch.cat([h(idx) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
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
    def __init__(self, vocab_size, tokenizer):
        super().__init__()
        self.device = get_device()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.block_count = block_count
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embed_dim, device=self.device)
        self.pos_embedding_table = nn.Embedding(self.block_size, self.embed_dim, device=self.device)
        self.blocks = nn.Sequential(*[Block() for _ in range(self.block_count)])
        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
        self.ln_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.token_embedding_table(x)
        pos = self.pos_embedding_table(torch.arange(T, device=x.device))
        x = x + pos
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
    def generate(self, idx):
        self.eval()
        for _ in range(block_size):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            next_token = idx_next[0, 0].item()
            if self.tokenizer.is_eos(next_token):
                break

            idx = torch.cat([idx, idx_next], dim=1)

        self.train()

        return idx
